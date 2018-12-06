# coding: utf-8

"""
The ``fine_tune.py`` file is used to continue training (or `fine-tune`) a model on a `different
dataset` than the one it was originally trained on.  It requires a saved model archive file, a path
to the data you will continue training with, and a directory in which to write the results.

. code-block:: bash

   $ python fine_tune.py --help
    usage: fine_tune.py [-h] -s SERIALIZATION_DIR -c CONFIG_FILE_PATH -p
                        PRETRAINED_DIR -m PRETRAINED_MODEL_NAME

    optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization_dir SERIALIZATION_DIR
                            Directory in which to save the model and its logs.
    -c CONFIG_FILE_PATH, --config_file_path CONFIG_FILE_PATH
                            Path to parameter file describing the new multi-tasked
                            model to be fine-tuned.
    -p PRETRAINED_DIR, --pretrained_dir PRETRAINED_DIR
                            Directory in which was saved the pre-trained model.
    -m PRETRAINED_MODEL_NAME, --pretrained_model_name PRETRAINED_MODEL_NAME
                            Name of the weight file for the pretrained model to
                            fine-tune in the ``pretrained_dir``.
"""

import argparse
import itertools
import os
import json
import re
from copy import deepcopy
import torch
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

from hmtl.tasks import Task
from hmtl.training.multi_task_trainer import MultiTaskTrainer
from hmtl.common import create_and_set_iterators
from evaluate import evaluate
from train import train_model

from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.commands.train import create_serialization_dir
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import RegularizerApplicator

logger = logging.getLogger(__name__)


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
        help="Path to parameter file describing the new multi-tasked model to be fine-tuned.",
        type=str,
    )
    parser.add_argument(
        "-p", "--pretrained_dir", required=True, help="Directory in which was saved the pre-trained model.", type=str
    )
    parser.add_argument(
        "-m",
        "--pretrained_model_name",
        required=True,
        help="Name of the weight file for the pretrained model to fine-tune in the ``pretrained_dir``.",
        type=str,
    )
    args = parser.parse_args()

    params = Params.from_file(params_file=args.config_file_path)
    serialization_dir = args.serialization_dir
    create_serialization_dir(params, serialization_dir, False)

    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, "config.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    ### Instantiate tasks ###
    task_list = []
    task_keys = [key for key in params.keys() if re.search("^task_", key)]

    for key in task_keys:
        logger.info("Creating %s", key)
        task_params = params.pop(key)
        task_description = task_params.pop("task_description")
        task_data_params = task_params.pop("data_params")

        task = Task.from_params(params=task_description)
        task_list.append(task)

        _, _ = task.load_data_from_params(params=task_data_params)

    ### Load Vocabulary from files and save it to the new serialization_dir ###
    # PLEASE NOTE that here, we suppose that the vocabulary is the same for the pre-trained model
    # and the model to fine-tune. The most noticeable implication of this hypothesis is that the label specs
    # between the two datasets (for pre-training and for fine-tuning) are exactly the same.
    vocab = Vocabulary.from_files(os.path.join(args.pretrained_dir, "vocabulary"))
    logger.info("Vocabulary loaded from %s", os.path.join(args.pretrained_dir, "vocabulary"))

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))
    logger.info("Save vocabulary to file %s", os.path.join(serialization_dir, "vocabulary"))

    ### Load the data iterators for each task ###
    task_list = create_and_set_iterators(params=params, task_list=task_list, vocab=vocab)

    ### Load Regularizations	###
    regularizer = RegularizerApplicator.from_params(params.pop("regularizer", []))

    ### Create model ###
    model_params = params.pop("model")
    model = Model.from_params(vocab=vocab, params=model_params, regularizer=regularizer)

    logger.info("Loading the pretrained model from %s", os.path.join(args.pretrained_dir, args.pretrained_model_name))
    try:
        pretrained_model_state_path = os.path.join(args.pretrained_dir, args.pretrained_model_name)
        pretrained_model_state = torch.load(pretrained_model_state_path)
        model.load_state_dict(state_dict=pretrained_model_state)
    except:
        raise ConfigurationError(
            "It appears that the configuration of the pretrained model and "
            "the model to fine-tune are not compatible. "
            "Please check the compatibility of the encoders and taggers in the "
            "config files."
        )

    ### Create multi-task trainer ###
    multi_task_trainer_params = params.pop("multi_task_trainer")
    trainer = MultiTaskTrainer.from_params(
        model=model, task_list=task_list, serialization_dir=serialization_dir, params=multi_task_trainer_params
    )

    ### Launch training ###
    metrics = train_model(multi_task_trainer=trainer, recover=False)
    if metrics is not None:
        logging.info("Fine-tuning is finished ! Let's have a drink. It's on the house !")
