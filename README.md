# Information Retrieval (IR)

[![License](https://img.shields.io/github/license/IgorAugust0/information-retrieval)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12.4%2B-blue)](https://www.python.org/downloads/)
[![NLTK](https://img.shields.io/badge/nltk-3.8.1-blue)](https://www.nltk.org/)
[![PrettyTable](https://img.shields.io/badge/prettytable-3.10.2-blue)](https://pypi.org/project/prettytable/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.9.1-blue)](https://matplotlib.org/)
[![TQDM](https://img.shields.io/badge/tqdm-4.66.4-blue)](https://tqdm.github.io/)

This repository contains three Python scripts implementing different models for Information Retrieval: Boolean Model, Vector Space Model, and an Evaluation script. Each script represents a different approach to Information Retrieval and they can be used to understand the basics of this field. They were made during the Information Retrieval "Organização e Recuperação da Informação (ORI)" [course](https://github.com/IgorAugust0/ORI) at the Federal University of Uberlândia (UFU).

## Programs

| Program | Description |
| --- | --- |
| [Boolean Model](1_boolean_model/base_samba/boolean_model.py) [(Instructions)](1_boolean_model/README.md) | Implements a Boolean Model for Information Retrieval, representing documents as sets of terms and queries as Boolean expressions of terms.  |
| [Vector Space Model](2_vector_space_model/base_samba/vsm.py) [(Instructions)](2_vector_space_model/README.md)| Implements a Vector Space Model for Information Retrieval, representing documents and queries as vectors in a high-dimensional space. |
| [Evaluation](3_evaluation/evaluation.py) [(Instructions)](3_evaluation/README.md)| This script is used to evaluate the performance of an Information Retrieval system. It calculates precision and recall scores for each query and generates plots for individual query results and the average scores of all queries. |

## Dependencies

These scripts require Python 3 and the following Python libraries installed:

- [**nltk**](https://www.nltk.org/install.html) (for both Boolean Model and Vector Space Model)
- [**prettytable**](https://pypi.org/project/prettytable/) (for the Boolean Model)
- [**matplotlib**](https://matplotlib.org/stable/install/index.html) (for the Evaluation script)
- [**tqdm**](https://tqdm.github.io/) (for all scripts, but it's likely that you already have it installed)

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

> you can use the command `pip freeze | %{$_.split('==')[0]} | %{pip install --upgrade $_}` to update all packages on Windows

## How to Run

To run the scripts, you need to pass the necessary files as command line arguments. Please refer to the individual scripts for more details.

## License

This program is provided under the [MIT License](LICENSE). Feel free to use, modify, and distribute it.
