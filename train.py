# coding: utf-8

"""
The ``train.py`` file can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python train.py --help
    usage: train.py [-h] -s SERIALIZATION_DIR -c CONFIG_FILE_PATH [-r]

    optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization_dir SERIALIZATION_DIR
                            Directory in which to save the model and its logs.
    -c CONFIG_FILE_PATH, --config_file_path CONFIG_FILE_PATH
                            Path to parameter file describing the multi-tasked
                            model to be trained.
    -r, --recover         Recover a previous training from the state in
                            serialization_dir.
"""

import argparse
import itertools
import os
import json
import re
from copy import deepcopy
import torch
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

from hmtl.tasks import Task
from hmtl.training.multi_task_trainer import MultiTaskTrainer
from hmtl.common import create_and_set_iterators
from evaluate import evaluate

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.commands.train import create_serialization_dir
from allennlp.common.params import Params
from allennlp.nn import RegularizerApplicator

logger = logging.getLogger(__name__)


def tasks_and_vocab_from_params(params: Params, serialization_dir: str) -> Tuple[List[Task], Vocabulary]:
    """
    Load each of the tasks in the model from the ``params`` file
    and load the datasets associated with each of these task.
    Create the vocavulary from ``params`` using the concatenation of the ``datasets_for_vocab_creation``
    from each of the task specific dataset.
    
    Parameters
    ----------
    params: ``Params``
        A parameter object specifing an experiment.
    serialization_dir: ``str``
        Directory in which to save the model and its logs.
    Returns
    -------
    task_list: ``List[Task]``
        A list containing the tasks of the model to train.
    vocab: ``Vocabulary``
        The vocabulary fitted on the datasets_for_vocab_creation.
    """
    ### Instantiate the different tasks ###
    task_list = []
    instances_for_vocab_creation = itertools.chain()
    datasets_for_vocab_creation = {}
    task_keys = [key for key in params.keys() if re.search("^task_", key)]

    for key in task_keys:
        logger.info("Creating %s", key)
        task_params = params.pop(key)
        task_description = task_params.pop("task_description")
        task_data_params = task_params.pop("data_params")

        task = Task.from_params(params=task_description)
        task_list.append(task)

        task_instances_for_vocab, task_datasets_for_vocab = task.load_data_from_params(params=task_data_params)
        instances_for_vocab_creation = itertools.chain(instances_for_vocab_creation, task_instances_for_vocab)
        datasets_for_vocab_creation[task._name] = task_datasets_for_vocab

    ### Create and save the vocabulary ###
    for task_name, task_dataset_list in datasets_for_vocab_creation.items():
        logger.info("Creating a vocabulary using %s data from %s.", ", ".join(task_dataset_list), task_name)

    logger.info("Fitting vocabulary from dataset")
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}), instances_for_vocab_creation)

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    logger.info("Vocabulary saved to %s", os.path.join(serialization_dir, "vocabulary"))

    return task_list, vocab


def train_model(multi_task_trainer: MultiTaskTrainer, recover: bool = False) -> Dict[str, Any]:
    """
    Launching the training of the multi-task model.
    
    Parameters
    ----------
    multi_task_trainer: ``MultiTaskTrainer``
        A trainer (similar to allennlp.training.trainer.Trainer) that can handle multi-task training.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
                    
    Returns
    -------
    metrics: ``Dict[str, Any]
        The different metrics summarizing the training of the model.
        It includes the validation and test (if necessary) metrics.
    """
    ### Train the multi-task model ###
    metrics = multi_task_trainer.train(recover=recover)

    task_list = multi_task_trainer._task_list
    serialization_dir = multi_task_trainer._serialization_dir
    model = multi_task_trainer._model

    ### Evaluate the model on test data if necessary ###
    # This is a multi-task learning framework, the best validation metrics for one task are not necessarily
    # obtained from the same epoch for all the tasks, one epoch begin equal to N forward+backward passes,
    # where N is the total number of batches in all the training sets.
    # We evaluate each of the best model for each task (based on the validation metrics) for all the other tasks (which have a test set).
    for task in task_list:
        if not task._evaluate_on_test:
            continue

        logger.info("Task %s will be evaluated using the best epoch weights.", task._name)
        assert (
            task._test_data is not None
        ), "Task {} wants to be evaluated on test dataset but no there is no test data loaded.".format(task._name)

        logger.info("Loading the best epoch weights for task %s", task._name)
        best_model_state_path = os.path.join(serialization_dir, "best_{}.th".format(task._name))
        best_model_state = torch.load(best_model_state_path)
        best_model = model
        best_model.load_state_dict(state_dict=best_model_state)

        test_metric_dict = {}

        for pair_task in task_list:
            if not pair_task._evaluate_on_test:
                continue

            logger.info("Pair task %s is evaluated with the best model for %s", pair_task._name, task._name)
            test_metric_dict[pair_task._name] = {}
            test_metrics = evaluate(
                model=best_model,
                task_name=pair_task._name,
                instances=pair_task._test_data,
                data_iterator=pair_task._data_iterator,
                cuda_device=multi_task_trainer._cuda_device,
            )

            for metric_name, value in test_metrics.items():
                test_metric_dict[pair_task._name][metric_name] = value

        metrics[task._name]["test"] = deepcopy(test_metric_dict)
        logger.info("Finished evaluation of task %s.", task._name)

    ### Dump validation and possibly test metrics ###
    metrics_json = json.dumps(metrics, indent=2)
    with open(os.path.join(serialization_dir, "metrics.json"), "w") as metrics_file:
        metrics_file.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    return metrics


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--serialization_dir", required=True, help="Directory in which to save the model and its logs.", type=str
    )
    parser.add_argument(
        "-c",
        "--config_file_path",
        required=True,
        help="Path to parameter file describing the multi-tasked model to be trained.",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--recover",
        action="store_true",
        default=False,
        help="Recover a previous training from the state in serialization_dir.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        required=False,
        help='Overwrite the output directory if it exists'
    )
    args = parser.parse_args()

    params = Params.from_file(params_file=args.config_file_path)
    serialization_dir = args.serialization_dir
    create_serialization_dir(params, serialization_dir, args.recover, args.force)

    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, "config.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    ### Instantiate the different tasks from the param file, load datasets and create vocabulary ###
    tasks, vocab = tasks_and_vocab_from_params(params=params, serialization_dir=serialization_dir)

    ### Load the data iterators for each task ###
    tasks = create_and_set_iterators(params=params, task_list=tasks, vocab=vocab)

    ### Load Regularizations ###
    regularizer = RegularizerApplicator.from_params(params.pop("regularizer", []))

    ### Create model ###
    model_params = params.pop("model")
    model = Model.from_params(vocab=vocab, params=model_params, regularizer=regularizer)

    ### Create multi-task trainer ###
    multi_task_trainer_params = params.pop("multi_task_trainer")
    trainer = MultiTaskTrainer.from_params(
        model=model, task_list=tasks, serialization_dir=serialization_dir, params=multi_task_trainer_params
    )

    ### Launch training ###
    metrics = train_model(multi_task_trainer=trainer, recover=args.recover)
    if metrics is not None:
        logging.info("Training is finished ! Let's have a drink. It's on the house !")
