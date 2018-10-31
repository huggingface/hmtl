# coding: utf-8

from typing import List
from allennlp.common import Params
from allennlp.commands.train import datasets_from_params
from allennlp.data.iterators import DataIterator
from allennlp.common.checks import ConfigurationError

class Task():
    """
    A class to encapsulate the necessary informations (and datasets)
    about each task.
    
    Parameters
    ----------
    name : ``str``, required
        The name of the task.
    validation_metric_name : ``str``, required
        The name of the validation metric to use to monitor training
        and select the best epoch.
    validation_metric_decreases : ``bool``, required
        Whether or not the validation metric should decrease for improvement.
    evaluate_on_test : ``bool`, optional (default = False)
        Whether or not the task should be evaluated on the test set at the end of the training.
    """
    def __init__(self,
                name: str,
                validation_metric_name: str,
                validation_metric_decreases: bool,
                evaluate_on_test: bool = False) -> None:
        self._name = name
        
        self._train_data = None
        self._validation_data = None
        self._test_data = None
        self._evaluate_on_test = evaluate_on_test
        
        self._val_metric = validation_metric_name
        self._val_metric_decreases = validation_metric_decreases
        
        self._data_iterator = None

        
    def set_data_iterator(self,
                    data_iterator: DataIterator):
        if data_iterator is not None:
            self._data_iterator = data_iterator
        else:
            ConfigurationError(f"data_iterator cannot be None in set_iterator - Task name: {self._name}")	

        
    def load_data_from_params(self, 
                            params: Params):
        all_datasets = datasets_from_params(params)
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))
        
        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")
        
        instances_for_vocab_creation = (instance for key, dataset in all_datasets.items() 
                                        for instance in dataset 
                                        if key in datasets_for_vocab_creation)
        
        self._instances_for_vocab_creation = instances_for_vocab_creation
        self._datasets_for_vocab_creation = datasets_for_vocab_creation
        
        if 'train' in all_datasets.keys():
            self._train_data = all_datasets["train"]
            self._tr_instances = sum(1 for e in self._train_data) # This is horrible if lazy iterator (Iterable)
        if 'validation' in all_datasets.keys():
            self._validation_data = all_datasets["validation"]
            self._val_instances = sum(1 for e in self._validation_data) # This is horrible if lazy iterator (Iterable)
        if 'test' in all_datasets.keys():
            self._test_data = all_datasets["test"]
            self._test_instances = sum(1 for e in self._test_data) # This is horrible if lazy iterator (Iterable)
        
        # If trying to evaluate on test set, make sure the dataset is loaded	
        if self._evaluate_on_test:
            assert self._test_data is not None
            
        #return instances_for_vocab_creation, datasets_for_vocab_creation, all_datasets
        return instances_for_vocab_creation, datasets_for_vocab_creation
    
    @classmethod	
    def from_params(cls, params: Params) -> "Task":
        task_name = params.pop("task_name", "ner")
        validation_metric_name = params.pop("validation_metric_name", "f1-measure-overall")
        validation_metric_decreases = params.pop_bool("validation_metric_decreases", False)
        evaluate_on_test = params.pop_bool("evaluate_on_test", False) 
        
        params.assert_empty(cls.__name__)
        return cls(name = task_name,
                validation_metric_name = validation_metric_name,
                validation_metric_decreases = validation_metric_decreases,
                evaluate_on_test = evaluate_on_test)