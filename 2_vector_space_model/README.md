# Vector Space Model Program

This Python program implements a Vector Space Model for information retrieval, allowing users to evaluate queries against a collection of documents. The Vector Space Model is a mathematical model used to represent text documents as vectors in a multidimensional space, facilitating similarity calculations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Functionality](#functionality)
- [How to Run](#how-to-run)
- [Command-line Arguments](#command-line-arguments)
- [Output](#output)
- [License](#license)

## Installation

Before using the program, ensure that the required NLTK package is installed. The program will attempt to install it if not already present. Otherwise, you can install it manually by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare Data:**
   - Create a base file (`base.txt`) containing a list of document filenames, with one filename per line.
   - Ensure the documents are present and accessible.

2. **Create Query File:**
   - Create a query file (`query.txt`) containing the query you want to evaluate.

3. **Run the Program:**
   - Execute the program with the following command:

     ```bash
     python vsm.py base.txt query.txt
     ```

## Dependencies

The program relies on the following dependencies:

- [NLTK](https://www.nltk.org/): Natural Language Toolkit for tokenization, stopword removal, and stemming.

## Functionality

The program performs the following tasks:

1. **Initialization:**
   - Checks for and installs the NLTK package.

2. **Document Preprocessing:**
   - Tokenizes, removes stopwords and punctuation, and performs stemming on the documents.

3. **Inverted Index Creation:**
   - Builds an inverted index from the preprocessed documents.

4. **Query Evaluation:**
   - Utilizes the Vector Space Model to calculate the similarity between the query and each document.

5. **Output:**
   - Saves weights of each term in each document, the inverted index, and the response to separate files.

6. **Execution Time Measurement:**
   - Measures and displays the total execution time.

## How to Run

1. Ensure Python is installed on your system.

2. Install the required dependencies (NLTK) using the provided installation instructions.

3. Prepare the input files: `base.txt` (document filenames) and `query.txt`.

4. Run the program from the command line as described in the [Usage](#usage) section.

## Command-line Arguments

The program expects two command-line arguments:

1. `base.txt`: Filename containing a list of document filenames.
2. `query.txt`: Filename containing the query for evaluation.

## Output

The program generates the following output files:

- `weights.txt`: Contains the weights of each term in each document.
- `index.txt`: Contains the inverted index.
- `response.txt`: Contains the ranked documents based on query similarity.

> **Note:** The output files are overwritten each time the program is run. To delete the output files and generate new ones, use the command `./delete.bat` (on Windows) or `./delete.sh` (on Linux) in your terminal.
---

## License

This program is provided under the [MIT License](LICENSE). Feel free to use, modify, and distribute it.
