[![Downloads](https://pepy.tech/badge/tarte-ai)](https://pepy.tech/project/tarte-ai)
[![PyPI Version](https://img.shields.io/pypi/v/tarte-ai)](https://pypi.org/project/tarte-ai/)
[![Python Version](https://img.shields.io/pypi/pyversions/tarte-ai)](https://pypi.org/project/tarte-ai/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Benchmark-yellow)](https://huggingface.co/inria-soda/tarte)
[![arXiv](https://img.shields.io/badge/arXiv-2505.14415-blue.svg)](https://arxiv.org/pdf/2505.14415)

# TARTE: Transformer Augmented Representation of Table Entries

![TARTE_outline](assets/TARTE_summary.jpg)

### NOTE: The repository is still in construction 🚧

This repository contains the implementation of **TARTE: Transformer Augmented Representation of Table Entries**.

TARTE is an easily-reusable pre-trained model that encodes data semantics across heterogeneous tables by pre-training from large knowledge bases. TARTE is a sibling work of [CARTE](https://github.com/soda-inria/carte), sharing many similarities, but with better pre-training and with more post-training paradigms.


> [!WARNING] <br>
> This library is currently in a phase of active development. All features are subject to change without prior notice. If you are interested in collaborating, please feel free to reach out by opening an issue or starting a discussion.


## Install

You can simply install TARTE from PyPI:

<pre>
pip install tarte-ai
</pre>

#### Post installation check
After a correct installation, you should be able to import the module without errors:

```python
import tarte_ai
```

## Colab Examples (Give it a test):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCrf27lfFQbEJgbdldNDdfvkYiLZNaVz?usp=sharing)
* TARTE Finetuning
* TARTE Featurizer
* TARTE Boosting
  


## Pre-training and reproducing the results from the paper.

Details will soon be updated.

## TARTE-AI reference

```
@article{kim2025table,
  title={Table Foundation Models: on knowledge pre-training for tabular learning},
  author={Kim, Myung Jun and Lefebvre, F{\'e}lix and Brison, Ga{\"e}tan and Perez-Lebel, Alexandre and Varoquaux, Ga{\"e}l},
  journal={arXiv preprint arXiv:2505.14415},
  year={2025}
}
```


