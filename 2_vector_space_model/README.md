# Vector Space Model

This Python program implements a Vector Space Model for information retrieval, allowing users to evaluate queries against a collection of documents. The Vector Space Model is a mathematical model used to represent text documents as vectors in a multidimensional space, facilitating similarity calculations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [How it Works?](#how-it-works)
- [Command-line Arguments](#command-line-arguments)
- [Output](#output)
- [License](#license)

## Installation

Before using the program, ensure that the required Natural Language Toolkit [NLTK](https://www.nltk.org/) package for tokenization, stopword removal, and stemming is installed. The program will attempt to install it if not already present. Otherwise, you can install it manually by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

> This file is located at the root of the repository.

## Usage

1. **Prepare Data:**

   - Create a base file (`base.txt`) containing a list of document filenames, with one filename per line.

     ```bash
     doc1.txt
     doc2.txt
     doc3.txt
     ```

     > Ensure the listed documents are present and accessible in the same directory as the program.

2. **Create Query File:**

   - Create a query file (`query.txt`) containing the query you want to evaluate.

3. **Run the Program:**

   - Execute the program with the following command:

     ```bash
     python vsm.py base.txt query.txt
     ```

## How It Works

The program executes the following steps:

### 1. Initialization

- Checks for and installs the **NLTK** package if not already present.

### 2. Document Preprocessing

- **Tokenization:** Splits the text into individual terms.
- **Cleaning:** Removes stopwords and punctuation.
- **Stemming:** Reduces terms to their root/base form.

### 3. Inverted Index Creation

- **Builds an inverted index** from the preprocessed documents. The index represents the relationship between terms and documents in the format `term:(document, weight)`, where the weight is computed using **TF-IDF** (Term Frequency-Inverse Document Frequency), as follows:

  $$\text{TF-IDF} = \text{TF} \times \text{IDF}$$

  - **Term Frequency (TF):**

    - Measures how frequently a term appears in a document, normalized by the total number of terms in the document.
    - Calculated as:
      - $1 + \log(\text{freq})$, if $\text{freq} > 0$
      - $0$, otherwise

  - **Inverse Document Frequency (IDF):**

    - Measures how important a term is by considering the number of documents containing the term
    - Helps in distinguishing common terms from rare ones.
    - Defined by the logarithm of the ratio of the total number of documents to the number of documents containing the term.
    - Calculated as:
      - $\log\left(\frac{N}{n}\right)$

  - **TF-IDF Weight:**

    - The product of TF and IDF.
    - Calculated as:
      - $\text{TF} \times \text{IDF}$

    Where:

    - $N$ is the total number of documents.
    - $n$ is the number of documents containing the term.

### 4. Query Evaluation

- **Preprocessing:** Applies the same preprocessing steps to the query, including handling logical operators (e.g., AND).
- **Similarity Calculation:** Evaluates similarity between the query and each document using the Vector Space Model and TF-IDF weighting. Similarity is computed as the cosine of the angle between document $d_j$ and query $q$, using:

  $$\text{Similarity} = \frac{\mathbf{v}_d \cdot \mathbf{v}_q}{\|\mathbf{v}_d\| \|\mathbf{v}_q\|}$$

- where $\mathbf{v}_d$ and $\mathbf{v}_q$ are the vectors for the document and query, respectively.

## Command-line Arguments

The program expects two command-line arguments:

1. `base.txt`: Filename containing a list of document filenames.
2. `query.txt`: Filename containing the query for evaluation.

## Output

The program generates the following output files:

- **Index (`index.txt`):** Contains the inverted index with terms and their corresponding documents and weights:

  ```bash
  am: 3,1 # term am appears in document 3 with weight 1
  cas: 1,1  2,4  3,3 # term cas appears in documents 1, 2, and 3 with weights 1, 4, and 3, respectively
  ...
  ```

- **Weights (`weights.txt`):** Contains documents and the weight of each term in the document:

  ```bash
  doc1.txt: term1, 0.1845 term2, 0.3010 # Document 1 with weights for terms 1 and 2
  doc2.txt: term1, 0.1625 term2, 0.6021 # Document 2 with weights for terms 1 and 2
  ```

- **Response (`response.txt`):** Lists the number of documents with a similarity greater than 0.001, showing the number of returned documents and their similarities, i.e., the ranked documents based on query similarity:

  ```bash
  3 # Number of documents with similarity > 0.001
  doc2.txt 0.9983 # Document 2 with similarity 0.9983
  doc3.txt 0.2031 # Document 3 with similarity 0.2031
  doc1.txt 0.1061 # Document 1 with similarity 0.1061
  ```

> **Note:** The output files are overwritten each time the program is runs. To delete the output files and generate new ones, use the command `./delete.bat` (on Windows) or `./delete.sh` (on Linux) in your terminal.

## License

This program is provided under the [MIT License](../LICENSE). Feel free to use, modify, and distribute it.
