# coding: utf-8

# A modified version of the trainer showcased in GLUE: https://github.com/nyu-mll/GLUE-baselines

import os
import math
import time
from copy import deepcopy
import random
import logging
import itertools
import shutil
from tensorboardX import SummaryWriter
import numpy as np

from typing import List, Optional, Dict, Any
from overrides import overrides

import torch
import torch.optim.lr_scheduler
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.nn.util import device_mapping
from allennlp.data.iterators import DataIterator
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import sparse_clip_norm, TensorboardWriter
from allennlp.models.model import Model

from hmtl.tasks import Task
from hmtl.training.multi_task_trainer import MultiTaskTrainer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@MultiTaskTrainer.register("sampler_multi_task_trainer")
class SamplerMultiTaskTrainer(MultiTaskTrainer):
    def __init__(
        self,
        model: Model,
        task_list: List[Task],
        optimizer_params: Params,
        lr_scheduler_params: Params,
        patience: Optional[int] = None,
        num_epochs: int = 20,
        serialization_dir: str = None,
        cuda_device: int = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        min_lr: float = 0.00001,
        no_tqdm: bool = False,
        summary_interval: int = 50,
        log_parameter_statistics: bool = False,
        log_gradient_statistics: bool = False,
        sampling_method: str = "proportional",
    ):

        if sampling_method not in ["uniform", "proportional"]:
            raise ConfigurationError(f"Sampling method ({sampling_method}) must be `uniform` or `proportional`.")

        self._sampling_method = sampling_method
        super(SamplerMultiTaskTrainer, self).__init__(
            model=model,
            task_list=task_list,
            optimizer_params=optimizer_params,
            lr_scheduler_params=lr_scheduler_params,
            patience=patience,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            min_lr=min_lr,
            no_tqdm=no_tqdm,
            summary_interval=summary_interval,
            log_parameter_statistics=log_parameter_statistics,
            log_gradient_statistics=log_gradient_statistics,
        )

    @overrides
    def train(self, recover: bool = False):
        """
        Train the different task_list, save the different checkpoints and metrics,
        and save the model at the end of training while logging the training details.
        
        The metrics through the training are stored in dictionaries with the following structure:
        
        all_metrics - Dict[str, str]
            task_name: val_metric

        metric_infos (Dict[])
            task_name (Dict[str, diverse]
                val_metric (str): name (str)
                hist (str): history_of_the_val_metric (List[float])
                stopped (str): training_is_stopped (bool)
                best (str): best_epoch_for_val_metric (Tuple(int, Dict))  

        all_tr_metrics (Dict[str, Dict[str, float]])
            task_name (Dict[str, float])
                metric_name (str): value (float)
                loss: value (float)		

        all_val_metrics (Dict[str, Dict[str, float]])
            task_name (Dict[str, float])
                metric_name (str): value (float)
                loss (str): value (float)
        
        Parameters
        ----------
        task_list: List[Task], required
            A list containing the tasks to train.
        params: Params, required
            Training parameters
        recover: bool, required
            Whether or not training should be recovered from a previous training.

        Returns
        -------
        return_dict: Dict
            A dictionary summarizing the training and the metrics for the best epochs for each task.
        """
        training_start_time = time.time()

        if recover:
            try:
                n_epoch, should_stop = self._restore_checkpoint()
                logger.info("Loaded model from checkpoint. Starting at epoch %d", n_epoch)
            except RuntimeError:
                raise ConfigurationError(
                    "Could not recover training from the checkpoint.  Did you mean to output to "
                    "a different serialization directory or delete the existing serialization "
                    "directory?"
                )
        else:
            n_epoch, should_stop = 0, False

            ### Store all the necessary informations and attributes about the tasks ###
            task_infos = {task._name: {} for task in self._task_list}
            for task_idx, task in enumerate(self._task_list):
                task_info = task_infos[task._name]

                # Store statistiscs on training and validation batches
                data_iterator = task._data_iterator
                n_tr_batches = data_iterator.get_num_batches(task._train_data)
                n_val_batches = data_iterator.get_num_batches(task._validation_data)
                task_info["n_tr_batches"] = n_tr_batches
                task_info["n_val_batches"] = n_val_batches

                # Create counter for number of batches trained during the whole
                # training for this specific task
                task_info["total_n_batches_trained"] = 0

                task_info["last_log"] = time.time()  # Time of last logging
            self._task_infos = task_infos

            ### Bookkeeping the validation metrics ###
            metric_infos = {
                task._name: {
                    "val_metric": task._val_metric,
                    "hist": [],
                    "is_out_of_patience": False,
                    "min_lr_hit": False,
                    "best": (-1, {}),
                }
                for task in self._task_list
            }
            self._metric_infos = metric_infos

        ### Write log ###
        total_n_tr_batches = 0  # The total number of training batches across all the datasets.
        for task_name, info in self._task_infos.items():
            total_n_tr_batches += info["n_tr_batches"]
            logger.info("Task %s:", task_name)
            logger.info("\t%d training batches", info["n_tr_batches"])
            logger.info("\t%d validation batches", info["n_val_batches"])

        ### Create the training generators/iterators tqdm ###
        self._tr_generators = {}
        for task in self._task_list:
            data_iterator = task._data_iterator
            tr_generator = data_iterator(task._train_data, num_epochs=None)
            self._tr_generators[task._name] = tr_generator

        ### Create sampling probability distribution ###
        if self._sampling_method == "uniform":
            sampling_prob = [float(1 / self._n_tasks)] * self._n_tasks
        elif self._sampling_method == "proportional":
            sampling_prob = [float(info["n_tr_batches"] / total_n_tr_batches) for info in self._task_infos.values()]

        ### Enable gradient clipping ###
        # Only if self._grad_clipping is specified
        self._enable_gradient_clipping()

        ### Setup is ready. Training of the model can begin ###
        logger.info("Set up ready. Beginning training/validation.")

        ### Begin Training of the model ###
        while not should_stop:
            # Train one epoch (training pass + validation pass)

            self._model.train()  # Set the model to "train" mode.

            ### Log Infos: current epoch count and CPU/GPU usage ###
            logger.info("")
            logger.info("Epoch %d/%d - Begin", n_epoch, self._num_epochs - 1)
            logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
            for gpu, memory in gpu_memory_mb().items():
                logger.info(f"GPU {gpu} memory usage MB: {memory}")

            logger.info("Training - Begin")

            ### Reset training and trained batches counter before new training epoch ###
            for _, task_info in self._task_infos.items():
                task_info["tr_loss_cum"] = 0.0
                task_info["n_batches_trained_this_epoch"] = 0
            all_tr_metrics = {}  # BUG TO COMPLETE COMMENT TO MAKE IT MORE CLEAR

            ### Start training epoch ###
            epoch_tqdm = tqdm.tqdm(range(total_n_tr_batches), total=total_n_tr_batches)
            for _ in epoch_tqdm:
                task_idx = np.argmax(np.random.multinomial(1, sampling_prob))
                task = self._task_list[task_idx]
                task_info = self._task_infos[task._name]

                ### One forward + backward pass ###

                # Call next batch to train
                batch = next(self._tr_generators[task._name])
                task_info["n_batches_trained_this_epoch"] += 1

                # Load optimizer
                optimizer = self._optimizers[task._name]
                optimizer.zero_grad()

                # Get the loss for this batch
                output_dict = self._forward(tensor_batch=batch, task=task, for_training=True)
                assert "loss" in output_dict, "Model must return a dict containing a 'loss' key"
                loss = output_dict["loss"]
                loss.backward()
                task_info["tr_loss_cum"] += loss.item()

                # Gradient rescaling if self._grad_norm is specified
                self._rescale_gradients()

                # Take an optimization step
                optimizer.step()

                ### Get metrics for all progress so far, update tqdm, display description ###
                task_metrics = self._get_metrics(task=task)
                task_metrics["loss"] = float(
                    task_info["tr_loss_cum"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
                )
                description = self._description_from_metrics(task_metrics)
                epoch_tqdm.set_description(task._name + ", " + description)

                ### Tensorboard logging: Training detailled metrics, parameters and gradients ###
                if self._global_step % self._summary_interval == 0:
                    # Metrics
                    for metric_name, value in task_metrics.items():
                        self._tensorboard.add_train_scalar(
                            name="training_details/" + task._name + "/" + metric_name,
                            value=value,
                            global_step=self._global_step,
                        )
                    # Parameters and Gradients
                    for param_name, param in self._model.named_parameters():
                        if self._log_parameter_statistics:
                            self._tensorboard.add_train_scalar(
                                name="parameter_mean/" + param_name,
                                value=param.data.mean(),
                                global_step=self._global_step,
                            )
                            self._tensorboard.add_train_scalar(
                                name="parameter_std/" + param_name,
                                value=param.data.std(),
                                global_step=self._global_step,
                            )
                        if param.grad is None:
                            continue
                        if self._log_gradient_statistics:
                            self._tensorboard.add_train_scalar(
                                name="grad_mean/" + param_name,
                                value=param.grad.data.mean(),
                                global_step=self._global_step,
                            )
                            self._tensorboard.add_train_scalar(
                                name="grad_std/" + param_name,
                                value=param.grad.data.std(),
                                global_step=self._global_step,
                            )
                self._global_step += 1

            ### Bookkeeping all the training metrics for all the tasks on the training epoch that just finished ###
            for task in self._task_list:
                task_info = self._task_infos[task._name]

                task_info["total_n_batches_trained"] += task_info["n_batches_trained_this_epoch"]
                task_info["last_log"] = time.time()

                task_metrics = self._get_metrics(task=task, reset=True)
                if task._name not in all_tr_metrics:
                    all_tr_metrics[task._name] = {}
                for name, value in task_metrics.items():
                    all_tr_metrics[task._name][name] = value
                all_tr_metrics[task._name]["loss"] = float(
                    task_info["tr_loss_cum"] / (task_info["n_batches_trained_this_epoch"] + 0.000_000_01)
                )

                # Tensorboard - Training metrics for this epoch
                self._tensorboard.add_train_scalar(
                    name="training_proportions/" + task._name,
                    value=task_info["n_batches_trained_this_epoch"],
                    global_step=n_epoch,
                )
                for metric_name, value in all_tr_metrics[task._name].items():
                    self._tensorboard.add_train_scalar(
                        name="task_" + task._name + "/" + metric_name, value=value, global_step=n_epoch
                    )

            logger.info("Train - End")

            ### Begin validation of the model ###
            logger.info("Validation - Begin")
            all_val_metrics = {}

            self._model.eval()  # Set the model into evaluation mode

            for task_idx, task in enumerate(self._task_list):
                logger.info("Validation - Task %d/%d: %s", task_idx + 1, self._n_tasks, task._name)

                val_loss = 0.0
                n_batches_val_this_epoch_this_task = 0
                n_val_batches = self._task_infos[task._name]["n_val_batches"]
                scheduler = self._schedulers[task._name]

                # Create tqdm generator for current task's validation
                data_iterator = task._data_iterator
                val_generator = data_iterator(task._validation_data, num_epochs=1, shuffle=False)
                val_generator_tqdm = tqdm.tqdm(val_generator, total=n_val_batches)

                # Iterate over each validation batch for this task
                for batch in val_generator_tqdm:
                    n_batches_val_this_epoch_this_task += 1

                    # Get the loss
                    val_output_dict = self._forward(batch, task=task, for_training=False)
                    loss = val_output_dict["loss"]
                    val_loss += loss.item()

                    # Get metrics for all progress so far, update tqdm, display description
                    task_metrics = self._get_metrics(task=task)
                    task_metrics["loss"] = float(val_loss / n_batches_val_this_epoch_this_task)
                    description = self._description_from_metrics(task_metrics)
                    val_generator_tqdm.set_description(description)

                # Get task validation metrics and store them in all_val_metrics
                task_metrics = self._get_metrics(task=task, reset=True)
                if task._name not in all_val_metrics:
                    all_val_metrics[task._name] = {}
                for name, value in task_metrics.items():
                    all_val_metrics[task._name][name] = value
                all_val_metrics[task._name]["loss"] = float(val_loss / n_batches_val_this_epoch_this_task)

                # Tensorboard - Validation metrics for this epoch
                for metric_name, value in all_val_metrics[task._name].items():
                    self._tensorboard.add_validation_scalar(
                        name="task_" + task._name + "/" + metric_name, value=value, global_step=n_epoch
                    )

                ### Perform a patience check and update the history of validation metric for this task ###
                this_epoch_val_metric = all_val_metrics[task._name][task._val_metric]
                metric_history = self._metric_infos[task._name]["hist"]

                metric_history.append(this_epoch_val_metric)
                is_best_so_far, out_of_patience = self._check_history(
                    metric_history=metric_history,
                    cur_score=this_epoch_val_metric,
                    should_decrease=task._val_metric_decreases,
                )

                if is_best_so_far:
                    logger.info("Best model found for %s.", task._name)
                    self._metric_infos[task._name]["best"] = (n_epoch, all_val_metrics)
                if out_of_patience and not self._metric_infos[task._name]["is_out_of_patience"]:
                    self._metric_infos[task._name]["is_out_of_patience"] = True
                    logger.info("Task %s is out of patience and vote to stop the training.", task._name)

                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                scheduler.step(this_epoch_val_metric, n_epoch)

            logger.info("Validation - End")

            ### Print all training and validation metrics for this epoch ###
            logger.info("***** Epoch %d/%d Statistics *****", n_epoch, self._num_epochs - 1)
            for task in self._task_list:
                logger.info("Statistic: %s", task._name)
                logger.info(
                    "\tTraining - %s: %3d",
                    "Nb batches trained",
                    self._task_infos[task._name]["n_batches_trained_this_epoch"],
                )
                for metric_name, value in all_tr_metrics[task._name].items():
                    logger.info("\tTraining - %s: %3f", metric_name, value)
                for metric_name, value in all_val_metrics[task._name].items():
                    logger.info("\tValidation - %s: %3f", metric_name, value)
            logger.info("**********")

            ### Check to see if should stop ###
            stop_tr, stop_val = True, True

            for task in self._task_list:
                # task_info = self._task_infos[task._name]
                if self._optimizers[task._name].param_groups[0]["lr"] < self._min_lr:
                    logger.info("Minimum lr hit on %s.", task._name)
                    logger.info("Task %s vote to stop training.", task._name)
                    metric_infos[task._name]["min_lr_hit"] = True
                stop_tr = stop_tr and self._metric_infos[task._name]["min_lr_hit"]
                stop_val = stop_val and self._metric_infos[task._name]["is_out_of_patience"]

            if stop_tr:
                should_stop = True
                logging.info("All tasks hit minimum lr. Stopping training.")
            if stop_val:
                should_stop = True
                logging.info("All metrics ran out of patience. Stopping training.")
            if n_epoch >= self._num_epochs - 1:
                should_stop = True
                logging.info("Maximum number of epoch hit. Stopping training.")

            self._save_checkpoint(n_epoch, should_stop)

            ### Update n_epoch ###
            # One epoch = doing N (forward + backward) pass where N is the total number of training batches.
            n_epoch += 1

        ### Summarize training at the end ###
        logging.info("***** Training is finished *****")
        logging.info("Stopped training after %d epochs", n_epoch)
        return_metrics = {}
        for task_name, task_info in self._task_infos.items():
            nb_epoch_trained = int(task_info["total_n_batches_trained"] / task_info["n_tr_batches"])
            logging.info(
                "Trained %s for %d batches ~= %d epochs",
                task_name,
                task_info["total_n_batches_trained"],
                nb_epoch_trained,
            )
            return_metrics[task_name] = {
                "best_epoch": self._metric_infos[task_name]["best"][0],
                "nb_epoch_trained": nb_epoch_trained,
                "best_epoch_val_metrics": self._metric_infos[task_name]["best"][1],
            }

        training_elapsed_time = time.time() - training_start_time
        return_metrics["training_duration"] = time.strftime("%d:%H:%M:%S", time.gmtime(training_elapsed_time))
        return_metrics["nb_epoch_trained"] = n_epoch

        return return_metrics

    @classmethod
    def from_params(
        cls, model: Model, task_list: List[Task], serialization_dir: str, params: Params
    ) -> "SamplerMultiTaskTrainer":
        """ Generator multi-task trainer from parameters.  """

        optimizer_params = params.pop("optimizer")
        lr_scheduler_params = params.pop("scheduler")
        patience = params.pop_int("patience", 2)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = params.pop_int("cuda_device", -1)
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        min_lr = params.pop_float("min_lr", 0.00001)
        no_tqdm = params.pop_bool("no_tqdm", False)
        summary_interval = params.pop("sumarry_interval", 50)
        log_parameter_statistics = params.pop("log_parameter_statistics", False)
        log_gradient_statistics = params.pop("log_gradient_statistics", False)
        sampling_method = params.pop("sampling_method", "proportional")

        params.assert_empty(cls.__name__)
        return SamplerMultiTaskTrainer(
            model=model,
            task_list=task_list,
            optimizer_params=optimizer_params,
            lr_scheduler_params=lr_scheduler_params,
            patience=patience,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            min_lr=min_lr,
            no_tqdm=no_tqdm,
            summary_interval=summary_interval,
            log_parameter_statistics=log_parameter_statistics,
            log_gradient_statistics=log_gradient_statistics,
            sampling_method=sampling_method,
        )
