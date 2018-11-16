# HMTL (Hierarchical Multi-Task Learning model) 

[__A Hierarchical Multi-Task Approach for Learning Embeddings from Semantic Tasks__](https://arxiv.org/abs/1811.06031)\
Victor SANH, Thomas WOLF, Sebastian RUDER\
In the Proceedings of AAAI 2019

<img src="https://github.com/huggingface/jointmodelMD/blob/master/HMTL_architecture.png" alt="HMTL Architecture" width="350"/>

## About

HMTL is a Hierarchical Multi-Task Learning model which combine a set of four carefully selected semantic tasks (namely Named Entity Recoginition, Entity Mention Detection, Relation Extraction and Coreference Resolution). The model achieves state-of-the-art results on Named Entity Recognition, Entity Mention Detection and Relation Extraction. Using [SentEval](https://github.com/facebookresearch/SentEval), we show that as we move from the bottom to the top layers of the model, the model tend to learn more complex semantic representation.

For more details, we refer to our [paper](https://arxiv.org/abs/1811.06031).

We release here the code for _training_, _fine tuning_ and _evaluating_ HMTL. We hope that this code will be useful for building your own Multi-Task models (hierarchical or not). The code is written in __Python__ and powered by __Pytorch__.

## Dependecies and installation

The main dependencies are:
- [AllenNLP](https://github.com/allenai/allennlp)
- [PyTorch](https://pytorch.org/)
- [SentEval](https://github.com/facebookresearch/SentEval) (only for evaluating the embeddings)

The code works with __Python 3.6__. A stable version of the dependencies is listed in `requirements.txt`.

You can quickly setup a working environment by calling the script `./script/machine_setup.sh`. It installs Python 3.6, create a clean virtual environment, and install all the required dependencies (listed in `requirements.txt`). Please adapt the script depending on your needs.

## Example usage

We base our implementation on the [AllenNLP library](https://github.com/allenai/allennlp). For an introduction to this library, you should check [these tutorials](https://allennlp.org/tutorials).

An experiment is described in a _json_ configuration file (see `configs/*.json` for examples). The configuration file mainly describes the datasets to load, the model to create along with all the hyper-parameters of the model. 

Once you have set up your configuration file (and defined custom classes such `DatasetReaders` if needed), you can simply launch a training with the following command and arguments:

```bash
python train.py --config_file_path configs/hmtl_coref_conll.json --serialization_dir my_first_training
```

Once the training has started, you can simply follow the training in the terminal or open a [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) (please make sure you have installed Tensorboard and its Tensorflow dependecy before):

```bash
tensorboard --logdir my_first_training/log
```

## Evaluating the embeddings with SentEval

We used [SentEval](https://github.com/facebookresearch/SentEval) to assess the linguistic properties learned by the model. `hmtl_senteval.py` gives an example of how we can create an interface between SentEval and HMTL. It evaluates the linguistic properties learned by every layer of the hiearchy (shared based word embeddings and encoders).

## Data

To download the pre-trained embeddings we used in HMTL, you can simply launch the script `./script/data_setup.sh`.

We do not attach the datasets used to train HMTL for licensing reasons, but we invite you to collect them by yourself: [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19), [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/), and [ACE2005](https://catalog.ldc.upenn.edu/LDC2006T06). The configuration files expect the datasets to be placed in the `data/` folder.

## References

```
@article{
}
```
