# 🎮 Demo: HMTL (Hierarchical Multi-Task Learning model) 

## Introduction

This is a demonstration of our NLP system: HMTL is a neural model for resolving four fundamental tasks in NLP, namely *Named Entity Recognition*, *Entity Mention Detection*, *Relation Extraction* and *Coreference Resolution* using multi-task learning.

For a brief introduction to multi-task learning, you can refer to our blog post (LINK TO COME). Each of the four tasks considered is detailed in the following section. 

The web interface for the demo can be found here (LINK TO COME) for you to try and play with it. HMTL comes with the web visualization client if you prefer to run on your local machine.

<img src="https://github.com/huggingface/hmtl/blob/master/demo/HMTL_demo.png" alt="HMTL Demo" width="900"/>

## Setup

The web demo (LINK TO COME) is based on Python 3.6 and [AllenNLP](https://github.com/allenai/allennlp).

The easiest way to setup a clean and working environment with the necessary dependencies is to refer to the setup section in the [parent folder](https://github.com/huggingface/hmtl#dependecies-and-installation).
A few supplementary dependecies are listed in `requirements.txt`  and are required to run the demo.

We also release three pre-trained HMTL models on English corporas. The three models essentially differ by the size of the ELMo embeddings used and thus the size of the model. The bigger the model, the higher the performance:

| Model Name | NER (F1) | EMD (F1) | RE (F1) | CR(F1) | Description |
| --- | --- | --- | --- | --- | --- |
| conll_small_elmo | 85.73 | 83.51 | 58.40 | 62.85 | Small version of ELMo |
| conll_medium_elmo | 86.41 | 84.02 | 58.78 | 61.62 | Medium version of ELMo |
| conll_full_elmo _(default model)_ | 86.40 | 85.59 | 61.37 | 62.26 | Original version of ELMo |

To download the pre-trained models, please install [git lfs](https://git-lfs.github.com/) and do a `git lfs pull`. The weights of the model will be saved in the `model_dumps` folder.

## Description of the tasks

### Named Entity Recognition (NER)

_Named Entity Recognition_ aims at identifying and clasifying named entities (real-world object, such as persons, locations, etc. that can be denoted with a proper name).

[Homer Simpson]<sub>PERS</sub> lives in [Springfield]<sub>LOC</sub> with his wife and kids.

HMTL is trained on OntoNotes 5.0 and can recognized various types (18) of named entities: _PERSON_, _NORP_, _FAC_, _ORG_, _GPE_, _LOC_, etc.

### Entity Mention Detection (EMD)

_Entity Mention Detection_ aims at identifying and clasifying entity mentions (real-world object, such as persons, locations, etc. that are not necessarily denoted with a proper name).

[The men]<sub>PERS</sub> held on [the sinking vessel]<sub>VEH</sub> until [the ship]<sub>VEH</sub> was able to reach them from [Corsica]<sub>LOC</sub>.

HMTL can recognized different types of mentions: _PER_, _GPE_, _ORG_, _FAC_, _LOC_, _WEA_ and _VEH_.

### Relation Extraction (RE)
 
_Relation extraction_ aims at extracting the semantic relations between the mentions.
 
The different types of relation detectec by HMTL are the following:

| Shortname | Full Name | Description | Example |
| --- | --- | -- | -- |
| ART | Artifact | User-Owner-Inventor-Manufacturer | {Leonard de Vinci painted the Joconde., ARG1 = Leonard de Vinci, ARG2 = Joconde} |
| GEN-AFF | Gen-Affiliation | Citizen-Resident-Religion-Ethnicity, Org-Location | {The people of Iraq., ARG1 =  The people, ARG2 = Iraq} |
| ORG-AFF | Org-Affiliation | Employment, Founder, Ownership, Student-Alum, Sports-Affiliation, Investor-Shareholder, Membership | {Martin Geisler, ITV News, Safwan southern Iraq., ARG1 = Martin Geisler, ARG2 = ITV News} |
| PART-WHOLE | Part-whole | Artifact, Geographical, Subsidiary | {They could safeguard the fields in Iraq., ARG1 = the fields, ARG2 = Iraq} |
| PER-SOC | Person-social | Business, Family, Lasting-Personal | {Sean Flyn, son the famous actor Errol Flynn, ARG1 = son, ARG2 = Errol Flynn} |
| PHYS | Physical | Located, Near | {The two journalists worked from the hotel., ARG1 = the two journalists, ARG2 = the hotel} |

For more details, please refer to the [dataset release notes](https://pdfs.semanticscholar.org/3a9b/136ca1ab91592df36f148ef16095f74d009e.pdf).


### Coreference Resolution (CR)

In a text, two or more expressions can link to the same person or thing in the worl. _Coreference Resolution_ aims at finding the coreferent spans and cluster them.

[My mom]<sub>1</sub> tasted [the cake]<sub>2</sub>. [She]<sub>1</sub> liked [it]<sub>2</sub>.


## Using HMTL as a server

A simple example of server script for integrating HTML in a REST API is provided as an example in [server.py](https://github.com/huggingface/hmtl/blob/master/demo/server.py).
To launch a specific model (please make sure to be in a environment with all the dependencies before: `source .env/bin/activate`):

```bash
gunicorn -b:8000 'server:build_app(model_name="<model_name>")'
```

or simply launching the default (full) model:

```bash
gunicorn -b:8000 'server:build_app()'
```

You can then call then the model with the following command: `curl http://localhost:8000/jmd/?text=Barack%20Obama%20is%20the%20former%20president.`.

## References

```
@article{
}
```
