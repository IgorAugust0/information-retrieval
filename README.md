# Information Retrieval

This repository contains three Python scripts implementing different models for Information Retrieval: Boolean Model, Vector Space Model, and an Evaluation script. Each script represents a different approach to Information Retrieval and they can be used to understand the basics of this field.

## Table of Contents

| Program | Description |
| --- | --- |
| [Boolean Model](1_boolean_model/base_samba/boolean_model.py) | This script implements a simple Boolean Model for Information Retrieval. It represents documents as sets of terms and queries as Boolean expressions of terms. |
| [Vector Space Model](2_vector_space_model/base_samba/vsm.py) | This script implements a Vector Space Model for Information Retrieval. It represents documents and queries as vectors in a high-dimensional space. |
| [Evaluation](3_evaluation/evaluation.py) | This script is used to evaluate the performance of an Information Retrieval system. It calculates precision and recall scores for each query and generates plots for individual query results and the average scores of all queries. |

## Dependencies

These scripts require Python 3 and the following Python libraries installed:

- nltk (for both Boolean Model and Vector Space Model)
- prettytable (for the Boolean Model)
- matplotlib (for the Evaluation script)
- tqdm (for all scripts, but its likely that you already have it installed)

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## How to Run

To run the scripts, you need to pass the necessary files as command line arguments. Please refer to the individual scripts for more details.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
