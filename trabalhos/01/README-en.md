# Boolean Model Information Retrieval System

This is a simple information retrieval system that uses the Boolean model to evaluate queries and retrieve relevant documents. The system is implemented in Python and uses the NLTK library for text processing.

## How it Works

The system works by building an inverted index from a set of documents and then using the index to evaluate queries. The inverted index is a data structure that maps each term in the documents to the set of documents that contain that term. The Boolean model is used to evaluate queries by treating them as Boolean expressions and using the inverted index to find the documents that match the expression.

The system consists of the following steps:

1. **Load the documents**: The system loads a set of documents from a file. Each document is represented as a string.

2. **Build the inverted index**: The system builds an inverted index from the documents. The index is a dictionary that maps each term to the set of documents that contain that term. The index is built by performing the following steps for each document:

   1. Tokenize the document into words.
   2. Remove stop words and punctuation.
   3. Stem the remaining words using the RSLP stemmer.
   4. Add the document to the inverted index for each term that appears in the document.

3. **Load the query**: The system loads a query from a file. The query is represented as a string.

4. **Evaluate the query**: The system evaluates the query using the Boolean model. The query is treated as a Boolean expression and is evaluated using the inverted index. The system evaluates the expression using the following operators:

   - `&`: Boolean AND operator
   - `|`: Boolean OR operator
   - `!`: Boolean NOT operator

   The system evaluates the expression by performing the following steps:

   1. Tokenize the query into terms.
   2. Evaluate each sub-expression in the query recursively.
   3. Combine the results of the sub-expressions using the Boolean operators.

5. **Save and display the query results**: The system saves the query results to a file and displays them in the terminal. The results are represented as a list of documents that match the query.

## How to Use

To use the system, follow these steps:

1. Install the required dependencies by running `pip install -r requirements.txt` in your terminal.

2. Create a file `base.txt` containing the documents you want to search. Each document should be on a separate line.

3. Run the program by running `python modelo_booleano.py base.txt consulta.txt` in your terminal. Replace `consulta.txt` with the name of the file containing your query.

4. The program will display the results of the query in the terminal and save them to a file `resposta.txt`.

5. To delete the output files and generate new ones, use the command `./delete.bat` (on Windows) or `./delete.sh` (on Linux) in your terminal.

## Conclusion

The Boolean model is a simple but effective way to retrieve relevant documents from a set of documents. This system demonstrates how the model can be implemented using Python and the NLTK library. By following the steps outlined in this README, you can use the system to search for documents and retrieve relevant results.
