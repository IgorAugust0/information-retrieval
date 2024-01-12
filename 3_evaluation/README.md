# Precision and Recall Evaluation

This Python script is used to evaluate the performance of an information retrieval system. It calculates precision and recall scores for each query, computes precision at different recall levels, calculates the average scores of all queries, and generates plots for individual query results and the average scores of all queries. That is, it consists of calculating and plotting the graph and average precision and revocation for a reference collection. See the program orientations [pdf](trab3_ori_2023-1-en.pdf) for more details.

## Dependencies

This script requires Python 3 and the following Python libraries installed:

- matplotlib

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## How to Run

To run the script, you need to pass the reference file as a command line argument:

```bash
python evaluation.py reference.txt
```

Replace `reference.txt` with the name of your reference file.

## File Structure

The reference file should contain the number of reference queries, ideal responses, and system responses. The first line of the file should be the number of queries. The following lines should contain the ideal responses, one per line. The rest of the lines should contain the system responses.

## Functions

The script contains the following functions:

- `read_file(filename)`: Reads the reference file.
- `calculate_metrics(ideal_responses, system_responses)`: Calculates recall and precision scores for each query.
- `calculate_precision_recall(recall_scores, precision_scores)`: Computes precision at different recall levels for each query.
- `calculate_average(combined_scores, num_queries)`: Calculates the average scores of all queries and writes them to a file.
- `plot_results(results, average_scores, num_queries)`: Generates plots for individual query results and the average scores of all queries.

## Output

The script generates two types of plots:

- Individual query results: These plots show the precision and recall scores for each query.
- Average scores: This plot shows the average precision and recall scores for all queries.

The average scores are also written to a file named `average.txt`.

### License

This project is licensed under the MIT License - see the LICENSE.md file for details.
