# huggingNLP

## Introduction

HuggingNLP is an open-source library for resolving four fundamental tasks in NLP, namely Named Entity Recognition, Entity Mention Detection, Relation Extraction and Coreference Resolution.

## Setup

Adapt and launch `machine_setup.sh`.

## Launch a server

We also release three pre-trained HMTL models.

| Model Name | NER (F1) | EMD (F1) | RE (F1) | CR(F1 on CoNLL2012) |
| --- | --- | --- | --- | --- |
| conll_small_elmo | 85.73 | 83.51 | 58.40 | 62.85 |
| conll_medium_elmo | 86.41 | 84.02 | 58.78 | 61.62 |
| conll_full_elmo _(default model)_ | 86.40 | 85.59 | 61.37 | 62.26 |

To launch a specific model (please make sure to be in a valid environment before: `source pyhuggingnlp/bin/activate`):

```bash
gunicorn -w 2 -b:8000 'server:build_app(model_name="<model_name>", mode="<demo_or_prod>")'
```

or launching the default model and mode:

```bash
gunicorn -w 2 -b:8000 'server:build_app()'
```

## Description of the tasks

### Named Entity Recognition (NER)
### Entity Mention Detection (EMD)
### Relation Extraction (RE)
 
The different types of relation are the following:

| Shortname | Full Name | Description | Example |
| --- | --- | -- | -- |
| ART | Artifact | User-Owner-Inventor-Manufacturer | {Pablo Picasso painted the Joconde., ARG1 = Pablo Picasso, ARG2 = Joconde} |
| GEN-AFF | Gen-Affiliation | Citizen-Resident-Religion-Ethnicity, Org-Location | {The people of Iraq., ARG1 =  The people, ARG2 = Iraq} |
| ORG-AFF | Org-Affiliation | Employment, Founder, Ownership, Student-Alum, Sports-Affiliation, Investor-Shareholder, Membership | {Martin Geisler, ITV News, Safwan southern Iraq., ARG1 = Martin Geisler, ARG2 = ITV News} |
| PART-WHOLE | Part-whole | Artifact, Geographical, Subsidiary | {They could safeguard the fields in Iraq., ARG1 = the fields, ARG2 = Iraq} |
| PER-SOC | Person-social | Business, Family, Lasting-Personal | {Sean Flyn, son the famous actor Errol Flynn, ARG1 = son, ARG2 = Errol Flynn} |
| PHYS | Physical | Located, Near | {The two journalists worked from the hotel., ARG1 = the two journalists, ARG2 = the hotel} |

For more details, please refer to the [dataset release notes](https://pdfs.semanticscholar.org/3a9b/136ca1ab91592df36f148ef16095f74d009e.pdf).


### Coreference Resolution (CR)


## Performance of the model

| NER | EMD | RE | CR |
| --- | --- | -- | ---|
|     |     |    |    | 
