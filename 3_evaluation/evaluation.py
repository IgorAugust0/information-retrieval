#  Autor: Igor Augusto Reis Gomes   [12011BSI290]

import sys
from collections import defaultdict
import matplotlib.pyplot as plt


def read_file(filename):
    """Reads a file containing the n number of reference queries, ideal responses and system responses."""
    try:
        with open(filename, "r") as file:
            num_queries = int(next(file))
            ideal_responses = [set(next(file).split()) for _ in range(num_queries)]
            system_responses = [line.split() for line in file]
        return num_queries, ideal_responses, system_responses
    except IOError:
        print(f"Error opening file {filename}")
        sys.exit(1)


def calculate_metrics(ideal_responses, system_responses):
    """Calculates recall and precision scores for each query."""
    recall_scores = defaultdict(list)
    precision_scores = defaultdict(list)
    ideal_lengths = [len(response) for response in ideal_responses]

    for sys_index, sys_response in enumerate(system_responses):
        match_count = 0
        for sys_sub_index, sys_value in enumerate(sys_response):
            if sys_value in ideal_responses[sys_index]:
                match_count += 1
                recall_scores[sys_index].append((match_count / ideal_lengths[sys_index]) * 100)
                precision_scores[sys_index].append((match_count / (sys_sub_index + 1)) * 100)
    return recall_scores, precision_scores


def calculate_precision_recall(recall_scores, precision_scores):
    """Computes precision at different recall levels for each query."""
    combined_scores = defaultdict(list)
    results = []

    for recall, precision in zip(recall_scores, precision_scores):
        recall_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        precision_at_recall = defaultdict(list)
        recall_values = []

        for level in recall_levels:
            for index, recall_value in enumerate(recall_scores[recall]):
                if level <= recall_value:
                    precision_at_recall[level].append(precision_scores[precision][index])
                else:
                    precision_at_recall[level].append(0)
            recall_values.append(max(precision_at_recall[level]))

        for recall_level, recall_value in zip(recall_levels, recall_values):
            combined_scores[recall_level].append(round(recall_value))
        results.append((recall_levels, recall_values))
    return combined_scores, results


def calculate_average(combined_scores, num_queries):
    """Calculates the average scores of all queries and writes them to a file."""
    average_scores = [round(sum(combined_scores[score]) / num_queries) for score in combined_scores]

    with open("average.txt", "w", encoding="utf8") as file:
        for score in average_scores:
            file.write(str(score / 100) + " ")
    return average_scores


def plot_results(results, average_scores, num_queries):
    """Generates plots for individual query results and the average scores of all queries."""
    fig, axs = plt.subplots(1, num_queries, figsize=(4 * num_queries, 4))
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    markers = ["o", "v", "^", "<", ">", "s", "p", "*"]

    # Graphs with each query result
    for i, result in enumerate(results):
        ax = axs[i]
        ax.plot(result[0], result[1], color=colors[i % len(colors)], marker=markers[i % len(markers)], linestyle="-")
        ax.set_title("Query " + str(i + 1))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    fig.suptitle("Results")
    plt.tight_layout()
    plt.show()

    # Graph with mean (average) of all queries
    plt.figure(figsize=(6, 4))
    plt.plot([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], average_scores, color="b", marker="o", linestyle="-")
    plt.title("Queries Average")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluation.py reference.txt")
        sys.exit(1)
    filename = sys.argv[1]
    num_queries, ideal_responses, system_responses = read_file(filename)
    recall_scores, precision_scores = calculate_metrics(ideal_responses, system_responses)
    combined_scores, results = calculate_precision_recall(recall_scores, precision_scores)
    average_scores = calculate_average(combined_scores, num_queries)
    plot_results(results, average_scores, num_queries)


if __name__ == "__main__":
    main()
