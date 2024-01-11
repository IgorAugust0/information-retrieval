#  Autor: Igor Augusto Reis Gomes   [12011BSI290]

from collections import defaultdict
import math
import string
import sys
import time
import subprocess

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from typing import Any, Callable

PRINT = False # flag to enable printing in console


def install_nltk():
    """Function to install the nltk package using pip. If the package is already installed, it will be skipped."""
    try:
        import nltk
        print("\033[1;32;40m✅ nltk is installed!\033[0m")
    except ImportError:
        print("\033[1;31m⚠ nltk not found. Installing it...\033[0m")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
            import nltk  # Try to import again after installation
            print("\033[1;32;40m✅ nltk is now installed!\033[0m")
        except Exception as e:
            print(f"\033[1;31m⚠ Failed to install nltk: {e}\033[0m")
            sys.exit(1)


def download_nltk_resources(resources: list[str]):
    """Download the necessary resources from NLTK."""
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[1])


def initialize_nltk():
    """Initialize NLTK and download the necessary resources."""
    install_nltk()
    resources = ["tokenizers/punkt", "corpora/stopwords", "stemmers/rslp"]
    download_nltk_resources(resources)


def load_base(base_filename: str) -> list[str]:
    """Load document filenames and contents from a base file."""
    try:
        with open(base_filename, 'r', encoding='utf-8') as base_file:
            # Read each line from base_file and strip whitespace
            filenames: list[str] = [line.strip() for line in base_file]
            return [open(filename, 'r', encoding='utf-8').read() for filename in filenames]
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading base file: {e}")
        sys.exit(1)


def preprocess_text(text: str) -> list[str]:
    """Preprocess the text by tokenizing, removing stopwords and punctuation, and stemming."""
    # --- Tokenize the text ---
    lowered_text: str = text.lower()
    tokens: list[str] = word_tokenize(lowered_text)
    # --- Remove stopwords and punctuation ---
    stop_words = set(stopwords.words("portuguese")).union(set(string.punctuation)).union({'..', '...'})
    tokens = [token for token in tokens if token not in stop_words]
    # --- Stem the words ---
    stemmer = RSLPStemmer()
    return [stemmer.stem(token) for token in tokens]


def build_inverted_index(docs: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Build an inverted index from the preprocessed documents."""
    inverted_index = defaultdict(list)
    for doc_id, doc in enumerate(docs, start=1):
        tokens: list[str] = preprocess_text(doc)
        for token in tokens:
            inverted_index[token].append((doc_id, 1))

    # Updating frequencies
    for token, doc_list in inverted_index.items():
        freq_dict = defaultdict(int)
        for doc_id, _ in doc_list:
            freq_dict[doc_id] += 1
        inverted_index[token] = [(doc_id, freq) for doc_id, freq in freq_dict.items()]

    return dict(sorted(inverted_index.items()))


def display_inverted_index(inverted_index: dict[str, list[tuple[int, int]]]):
    """Display the inverted index in the terminal."""
    for term, doc_list in inverted_index.items():
        print(f"{term}: ", end="")
        for doc in doc_list:
            print(f"{doc[0]},{doc[1]} ", end="")
        print()


def save_inverted_index(inverted_index: dict[str, list[tuple[int, int]]]):
    """Save the inverted index to a file."""
    try:
        with open("indice.txt", encoding="utf8", mode="w") as file:
            for term, doc_list in inverted_index.items():
                file.write(f"{term}: ")
                for doc_id, freq in doc_list:
                    file.write(f"{doc_id},{freq}  ")
                file.write("\n")
    except IOError as e:
        print(f"Error saving inverted index: {e}")
        sys.exit(1)


def load_query(query_file: str) -> str:
    """Loads a query from a file and returns it a string."""
    try:
        with open(query_file, "r") as file:
            return file.read()
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading query file: {e}")
        sys.exit(1)


def term_frequency(inverted_index: dict[str, list[tuple[int, int]]]) -> dict[str, int]:
    """Calculates the term frequency of each term in the inverted index."""
    term_freqs: dict[str, int] = {}
    for term in inverted_index:
        term_freqs[term] = len(inverted_index[term])
    # Print the term weights in console if DEBUG is True
    if PRINT:
        print("\033[1;34m==== Term Frequencies ====\033[0m")
        for term, freq in term_freqs.items():
            print(f"{term}: {freq}")
    return term_freqs


def doc_frequency(inverted_index: dict[str, list[tuple[int, int]]]) -> dict[int, list[tuple[str, int]]]:
    """Calculates the document frequency of each document in the inverted index."""
    doc_freqs: dict[int, list[tuple[str, int]]] = {}
    # Calculate the document frequency for each term and add it to doc_freqs
    for term in inverted_index:
        for doc_id, freq in inverted_index[term]:
            if doc_id not in doc_freqs:
                doc_freqs[doc_id] = []
            doc_freqs[doc_id].append((term, freq))
    doc_freqs = dict(sorted(doc_freqs.items()))
    if PRINT:
        print("\n\033[1;34m==== Document Frequencies ====\033[0m")
        for doc_id in doc_freqs:
            print(f"Doc {doc_id}: {doc_freqs[doc_id]}")
    return doc_freqs


def term_weighting(inverted_index: dict[str, list[tuple[int, int]]]) -> dict[int, list[tuple[str, float]]]:
    """Calculates the term weighting of each term in the inverted index."""
    term_weights = defaultdict(list)
    # Calculate the total number of documents (N)
    n = len(set(doc_id for _, doc_list in inverted_index.items() for doc_id, _ in doc_list))
    # Calculate IDF for each term
    idf = {term: math.log(n / len(doc_list), 10) for term, doc_list in inverted_index.items()}

    # Calculate TF-IDF
    doc_freqs = doc_frequency(inverted_index)
    for doc_id, terms_list in doc_freqs.items():
        for term, freq in terms_list:
            tf = 1 + math.log(freq, 10) if freq > 0 else 0
            tf_idf = tf * idf[term]
            term_weights[doc_id].append((term, tf_idf))

    if PRINT:
        print("\n\033[1;34m==== Term Weights ====\033[0m")
        for doc_id in term_weights:
            print(f"Doc {doc_id}: {term_weights[doc_id]}")

    return term_weights


def vector_space_model(inverted_index: dict[str, list[tuple[int, int]]], query: str, base_filename: str) -> list[tuple[int, float]]:
    """Calculates the similarity between the query and each document using the vector space model."""
    document_weights = term_weighting(inverted_index)
    query_tokens = [token for token in preprocess_text(query) if token != '&']

    # Calculate IDF for query terms
    n = len(document_weights)
    query_term_weights = defaultdict(float)
    for token in query_tokens:
        if token in inverted_index:
            idf = math.log(n / len(inverted_index[token]), 10)
            tf = 1 + math.log(query_tokens.count(token), 10) if query_tokens.count(token) > 0 else 0
            query_term_weights[token] = tf * idf

    # Calculate the numerator part of similarity equation beforehand
    query_norm = math.sqrt(sum(weight**2 for weight in query_term_weights.values()))
    similarities = {}

    # Calculate the similarity between the query and each document
    for doc_id, doc_weights in document_weights.items():
        doc_norm = math.sqrt(sum(weight**2 for _, weight in doc_weights))
        internal_prod = sum(query_term_weights.get(weight, 0) * term for weight, term in doc_weights)
        similarities[doc_id] = internal_prod / (query_norm * doc_norm) if query_norm != 0 and doc_norm != 0 else 0

    save_weights(document_weights, base_filename)
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)


def read_doc_names(base_filename: str) -> list[str]:
    """Read the document names from the base file."""
    with open(base_filename, "r", encoding="utf-8") as base_file:
        return [line.strip() for line in base_file]


def save_weights(document_weights: dict[int, list[tuple[str, float]]], base_filename: str):
    """Save the weights of each term in each document to a file."""
    doc_names = read_doc_names(base_filename)

    with open("pesos.txt", "w", encoding="utf-8") as weights_file:
        for doc_id, weights in document_weights.items():
            index = doc_id - 1
            if index < len(doc_names):
                document_name = doc_names[index]
                weights_file.write(f"{document_name}: ")
                weights_file.write("  ".join(f"{term}, {weight}" for term, weight in weights if weight != 0))
                weights_file.write("\n")


def save_response(ranking: list[tuple[int, float]], base_filename: str):
    """Save the response to a file."""
    doc_names = read_doc_names(base_filename)
    # Filter the ranking to only include documents with similarity >= 0.001
    filtered_ranking = [(doc_id, similarity) for doc_id, similarity in ranking if similarity >= 0.001]

    # Save the response to a file
    with open("resposta.txt", "w", encoding="utf-8") as response_file:
        response_file.write(f"{len(filtered_ranking)}\n")
        for doc_id, similarity in filtered_ranking:
            index = doc_id - 1
            if index < len(doc_names):
                document_name = doc_names[index]
                response_file.write(f"{document_name} {similarity}\n")


def measure_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator function to measure the execution time of a function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        total_time: float = end_time - start_time
        print(f"Total execution time: {total_time:.2f} seconds\n")
        return result

    return wrapper


@measure_execution_time
def main(base_filename: str, query_filename: str):
    initialize_nltk()  # Initialize NLTK and download the necessary resources

    # --- Load the documents ---
    print("\033[1;37m📚 Loading documents...\033[0m")
    docs = load_base(base_filename)
    print(f"\033[1;32;40m✅ Loaded {len(docs)} documents!\033[0m")

    # --- Build the inverted index and save it to a file and display it in the terminal ---
    print("\033[1;36m🔍 Building inverted index...\033[0m")
    inverted_index = build_inverted_index(docs)
    print(f"\033[1;32;40m✅ Built inverted index!\033[0m")
    if PRINT:
        print("\n\033[1;34m==== Inverted Index ====\033[0m")
        display_inverted_index(inverted_index)
        save_inverted_index(inverted_index)
        print()

    # --- Load and read the query ---
    print("\033[1;34m📝 Loading query...\033[0m")
    query = load_query(query_filename)
    print("\033[1;35m🔎 Evaluating query...\033[0m")

    # --- Rank the documents ---
    ranked_docs = vector_space_model(inverted_index, query, base_filename)

    # --- Save the response ---
    save_response(ranked_docs, base_filename)

    # --- Print the execution information ---
    print("\n\033[1;34m==== Execution Information ====\033[0m")
    print(f"Base File: {base_filename}")
    print(f"Query File: {query_filename}")
    print("\n\033[1;33m✨ Done!\033[0m\n")


def parse_arguments():
    """Parse the command line arguments."""
    if len(sys.argv) != 3:
        print("\033[91m\n❌ ERROR: Invalid arguments!\n")
        print("\033[93mUsage: python nome_do_programa.py base.txt consulta.txt\n")
        print("\033[0m")
        sys.exit(1)
    return sys.argv[1], sys.argv[2]


if __name__ == "__main__":
    base_filename, query_filename = parse_arguments()
    main(base_filename, query_filename)
