# coding: utf-8

import os
import math
import time
from copy import deepcopy
import random
import logging
import itertools
import shutil
from tensorboardX import SummaryWriter

from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.optim.lr_scheduler
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.nn.util import device_mapping, move_to_device
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer import sparse_clip_norm, TensorboardWriter
from allennlp.models.model import Model
from allennlp.common.registrable import Registrable


from hmtl.tasks import Task

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class MultiTaskTrainer(Registrable):
    def __init__(self, 
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
                log_gradient_statistics: bool = False):
        """ 
        Parameters
        ----------
        model: ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        iterator: ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        patience: Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        num_epochs: int, optional (default = 20)
            Number of training epochs.
        serialization_dir: str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device: int, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm: float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : float, optional (default = None).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        no_tqdm : bool, optional (default=False)
            We use ``tqdm`` for logging, which will print a nice progress bar that updates in place
            after every batch.  This is nice if you're running training on a local shell, but can
            cause problems with log files from, e.g., a docker image running on kubernetes.  If
            ``no_tqdm`` is ``True``, we will not use tqdm, and instead log batch statistics using
            ``logger.info``.
        """
        self._model = model
        parameters_to_train = [(n, p) for n, p in self._model.named_parameters() if p.requires_grad]
        
        self._task_list = task_list
        self._n_tasks = len(self._task_list)
        
        self._optimizer_params = optimizer_params
        self._optimizers = {}
        self._lr_scheduler_params = lr_scheduler_params
        self._schedulers = {}
        for task in self._task_list:
            task_name = task._name
            self._optimizers[task_name] = Optimizer.from_params(model_parameters = parameters_to_train,
                                                                  params = deepcopy(optimizer_params))
            self._schedulers[task_name] = LearningRateScheduler.from_params(optimizer = self._optimizers[task_name],
                                                                            params = deepcopy(lr_scheduler_params))
        
        self._serialization_dir = serialization_dir
    
        self._patience = patience
        self._num_epochs = num_epochs
        self._cuda_device = cuda_device
        if self._cuda_device >= 0:
            check_for_gpu(self._cuda_device)
            self._model = self._model.cuda(self._cuda_device)
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._min_lr = min_lr

        self._task_infos = None
        self._metric_infos = None
        
        self._tr_generators = None
        self._no_tqdm = no_tqdm
        
        self._summary_interval = summary_interval  # num batches between logging to tensorboard
        self._log_parameter_statistics = log_parameter_statistics
        self._log_gradient_statistics = log_gradient_statistics
        self._global_step = 0
        train_log = SummaryWriter(os.path.join(self._serialization_dir, "log", "train"))
        validation_log = SummaryWriter(os.path.join(self._serialization_dir, "log", "validation"))
        self._tensorboard = TensorboardWriter(train_log = train_log, validation_log = validation_log)


    def train(self, 
            #tasks: List[Task], 
            #params: Params,
            recover: bool = False):
            
        raise NotImplementedError
    
    
    def _check_history(self, 
                    metric_history: List[float], 
                    cur_score: float, 
                    should_decrease: bool = False):
        '''
        Given a task, the history of the performance on that task,
        and the current score, check if current score is
        best so far and if out of patience.
        
        Parameters
        ----------
        metric_history: List[float], required
        cur_score: float, required
        should_decrease: bool, default = False
            Wheter or not the validation metric should increase while training.
            For instance, the bigger the f1 score is, the better it is -> should_decrease = False
            
        Returns
        -------
        best_so_far: bool
            Whether or not the current epoch is the best so far in terms of the speicified validation metric.
        out_of_patience: bool
            Whether or not the training for this specific task should stop (patience parameter).
        '''
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        out_of_patience = False
        if len(metric_history) > patience:
            if should_decrease:
                out_of_patience = max(metric_history[-patience:]) <= cur_score
            else:
                out_of_patience = min(metric_history[-patience:]) >= cur_score

        if best_so_far and out_of_patience: # then something is up
            print("Something is up")

        return best_so_far, out_of_patience
    
    
    def _forward(self, 
                tensor_batch: torch.Tensor, 
                for_training: bool = False,
                task:Task = None):
        if task is not None:
            tensor_batch = move_to_device(tensor_batch, self._cuda_device)
            output_dict = self._model.forward(task_name = task._name, tensor_batch = tensor_batch, for_training = for_training)
            if for_training:
                try:
                    loss = output_dict["loss"]
                    loss += self._model.get_regularization_penalty()
                except KeyError:
                    raise RuntimeError("The model you are trying to optimize does not contain a"
                                           " `loss` key in the output of model.forward(inputs).")
            return output_dict
        else:
            raise ConfigurationError("Cannot call forward through task `None`")
    
        
    def _get_metrics(self, 
                    task: Task, 
                    reset: bool = False):
        task_tagger = getattr(self._model, "_tagger_" + task._name)
        return task_tagger.get_metrics(reset)


    def _description_from_metrics(self, 
                                 metrics: Dict[str, float]):
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.4f" % (name, value) for name, value in metrics.items()]) + " ||"


    def _rescale_gradients(self) -> Optional[float]:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if self._grad_norm:
            parameters_to_clip = [p for p in self._model.parameters()
                                  if p.grad is not None]
            return sparse_clip_norm(parameters_to_clip, self._grad_norm)
        return None


    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)
    
                    
    def _save_checkpoint(self, 
                        epoch: int, 
                        should_stop: bool) -> None:
        """
        Save the current states (model, training, optimizers, metrics and tasks).
        
        Parameters
        ----------
        epoch: int, required.
            The epoch of training.
        should_stop: bool, required
            Wheter or not the training is finished.
        should_save_model: bool, optional (default = True)
            Whether or not the model state should be saved.
        """		
        ### Saving training state ###
        training_state = {"epoch": epoch, 
                        "should_stop": should_stop,
                        "metric_infos": self._metric_infos,
                        "task_infos": self._task_infos,
                        "schedulers": {},
                        "optimizers": {}}
                        
        if self._optimizers is not None:
            for task_name, optimizer in self._optimizers.items():
                training_state["optimizers"][task_name] = optimizer.state_dict()
        if self._schedulers is not None:
            for task_name, scheduler in self._schedulers.items():
                training_state["schedulers"][task_name] = scheduler.lr_scheduler.state_dict()
                
        training_path = os.path.join(self._serialization_dir, "training_state.th")
        torch.save(training_state, training_path)
        logger.info("Checkpoint - Saved training state to %s", training_path)
        
        
        ### Saving model state ###
        model_path = os.path.join(self._serialization_dir, "model_state.th")
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)
        logger.info("Checkpoint - Saved model state to %s", model_path)
        
        
        ### Saving best models for each task ###					 
        for task_name, infos in self._metric_infos.items():
            best_epoch, _ = infos["best"]
            if best_epoch == epoch:
                logger.info("Checkpoint - Best validation performance so far for %s task", task_name)
                logger.info("Checkpoint - Copying weights to '%s/best_%s.th'.", self._serialization_dir, task_name)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best_{}.th".format(task_name)))
    
    
    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = (self._serialization_dir is not None and
                           any("model_state" in x for x in os.listdir(self._serialization_dir)) and
                           any("training_state" in x for x in os.listdir(self._serialization_dir)))

        if not have_checkpoint:
            return None

        model_path = os.path.join(self._serialization_dir,
                                "model_state.th")
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state.th")

        return (model_path, training_state_path)
        
                
    def _restore_checkpoint(self):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count, optimizer state, a model state, a task state and
        a metric state. All are of which are serialized separately. 
        This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        Returns
        -------
        epoch: int, 
            The epoch at which to resume training.
        should_stop: bool
            Whether or not the training should already by stopped.
        """
        
        latest_checkpoint = self.find_latest_checkpoint()
        
        if not self._serialization_dir:
            raise ConfigurationError("`serialization_dir` not specified - cannot "
                                     "restore a model without a directory path.")
        if latest_checkpoint is None:
            raise ConfigurationError("Cannot restore model because one of"
                                    "`model_state.th` or `training_state.th` is not in directory path.")
        
        model_path, training_state_path = latest_checkpoint
        
        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        model_state = torch.load(model_path, map_location = device_mapping(-1))
        training_state = torch.load(training_state_path, map_location = device_mapping(-1))
        
        # Load model
        self._model.load_state_dict(model_state)
        logger.info("Checkpoint - Model loaded from %s", model_path)
        
        # Load optimizers
        for task_name, optimizers_state in training_state["optimizers"].items():
            self._optimizers[task_name].load_state_dict(optimizers_state)
        logger.info("Checkpoint - Optimizers loaded from %s", training_state_path)
        
        # Load schedulers
        for task_name, scheduler_state in training_state["schedulers"].items():
            self._schedulers[task_name].lr_scheduler.load_state_dict(scheduler_state)
        logger.info("Checkpoint - Learning rate schedulers loaded from %s", training_state_path)
        
        self._metric_infos = training_state["metric_infos"]
        self._task_infos = training_state["task_infos"]
        logger.info("Checkpoint - Task infos loaded from %s", training_state_path)
        logger.info("Checkpoint - Metric infos loaded from %s", training_state_path)
        
        n_epoch, should_stop = training_state["epoch"], training_state["should_stop"]
        
        return n_epoch + 1, should_stop


    @classmethod
    def from_params(cls,  
                    model: Model, 
                    task_list: List[Task],
                    serialization_dir: str,
                    params: Params) -> 'MultiTaskTrainer':
        """
        Static method that constructs the multi task trainer described by ``params``.
        """
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(model = model, 
                                            task_list = task_list,
                                            serialization_dir = serialization_dir,
                                            params = params)