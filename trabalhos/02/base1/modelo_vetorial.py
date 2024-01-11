#  Autor: Igor Augusto Reis Gomes   [12011BSI290]

import math
import pickle
import sys
import time
import subprocess

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tag import UnigramTagger
from nltk.tokenize import word_tokenize
from typing import Any, Callable

# you may need to run the following commands in your terminal to download nltk, tqdm:
# pip install nltk
# pip install tqdm


def install_nltk() -> None:
    """Function to install the nltk package using pip. If the package is already installed, it will be skipped."""
    try:
        import nltk

        print("\033[1;32;40m✅ nltk is installed!\033[0m")
    except ImportError:
        print("\033[1;31m⚠ nltk not found. Installing it...\033[0m")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        except Exception as e:
            print(f"\033[1;31m⚠ Failed to install nltk: {e}\033[0m")
            sys.exit(1)


def initialize_nltk() -> None:
    """Initialize NLTK and download the necessary resources."""
    install_nltk()
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/mac_morpho")
    except LookupError:
        nltk.download("mac_morpho")

    try:
        nltk.data.find("stemmers/rslp")
    except LookupError:
        nltk.download("rslp")


def load_tagger() -> UnigramTagger:
    """Load or create the NLTK tagger and save it to a binary file."""
    try:
        with open("etiquetador.bin", "rb") as file:
            tagger: UnigramTagger = pickle.load(file)
    except FileNotFoundError:
        tagged_sentences = nltk.corpus.mac_morpho.tagged_sents()
        tagger = UnigramTagger(tagged_sentences)
        with open("etiquetador.bin", "wb") as file:
            pickle.dump(tagger, file)
    return tagger


def load_documents(base_filename: str) -> tuple[list[str], dict[int, str]]:
    """Load document filenames and contents from a base file."""
    with open(base_filename, "r") as base_file:
        texts: list[str] = []
        docs: dict[int, str] = {}
        for i, line in enumerate(base_file):
            fields: list[str] = line.split()
            doc_filename: str = fields[0]
            docs.update({i + 1: doc_filename})
            with open(doc_filename, encoding="utf8", mode="r") as doc_file:
                texts.append(doc_file.read())
    return texts, docs


def preprocess_text(
    text: str,
    tagger: UnigramTagger,
    stemmer: RSLPStemmer,
    show_classifications: bool = False,
) -> dict[str, int]:
    """Preprocess the given text."""
    global global_classification_list
    global_classification_list = []
    word_list_final: list[str] = []
    sentence: str = text
    word_list: list[str] = word_tokenize(sentence)
    word_list = [val.lower() for val in word_list]
    classification_list: list[list[tuple[str, str]]] = []
    stop_words = stopwords.words("portuguese")

    for word in word_list:
        token = word_tokenize(word)
        classification = tagger.tag(token)
        classification_list.append(classification)

    stopword_classification_list: list[str] = []

    for tag in classification_list:
        for word, classification in tag:
            if (
                classification == "ART"
                or classification == "PREP"
                or classification == "KC"
                or classification == "KS"
            ):
                stopword_classification_list.append(word)

    removal_items = [".", "..", "...", "!", "?", ","]
    i = 0

    while i < len(word_list):
        if (
            word_list[i] in removal_items
            or word_list[i] in stop_words
            or word_list[i] in stopword_classification_list
        ):
            word_list.pop(i)
        else:
            i += 1

    for word in word_list:
        token = nltk.word_tokenize(word)
        classification = tagger.tag(token)
        global_classification_list.append(classification)
        word_list_final.append(stemmer.stem(word))

    word_count: dict[str, int] = {}
    for word in word_list_final:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] = word_count[word] + 1

    # Display classifications in the terminal if show_classifications is True
    if show_classifications:
        print("\033[1;35m==== Word Classification ====\033[0m")
        print_classifications(global_classification_list)
        print()

    return word_count


def print_classifications(global_classification_list):
    """Print the word-classification pairs in the given list."""
    for pair in global_classification_list:
        print(f"word - classification: {pair}")


def build_inverted_index(
    texts: list[str], tagger: UnigramTagger, stemmer: RSLPStemmer
) -> dict[str, dict[int, int]]:
    """Build an inverted index from the preprocessed documents."""
    inverted_index: dict[str, dict[int, int]] = {}
    for idx, text in enumerate(texts):
        word_count: dict[str, int] = preprocess_text(
            text, tagger, stemmer, show_classifications=True
        )
        doc_id: int = idx + 1
        for word in word_count:
            if word in inverted_index:
                inverted_index[word].update({doc_id: word_count[word]})
            else:
                inverted_index[word] = {doc_id: word_count[word]}

    sorted_inverted_index = dict(
        sorted(inverted_index.items(), key=lambda item: item[0])
    )
    return sorted_inverted_index


def display_inverted_index(inverted_index: dict[str, dict[int, int]]) -> None:
    """Display the inverted index in the terminal."""
    for term in inverted_index:
        print(f"{term}: ", end="")
        for doc_id in inverted_index[term]:
            print(f"{doc_id},{inverted_index[term][doc_id]} ", end="")
        print()
    return


def save_inverted_index(inverted_index: dict[str, dict[int, int]]) -> None:
    """Save the inverted index to a file."""
    with open("indice.txt", encoding="utf8", mode="w") as file:
        for term in sorted(inverted_index):
            file.write(term + ": ")
            file.write(
                " ".join(
                    f"{doc_id},{inverted_index[term][doc_id]}"
                    for doc_id in sorted(inverted_index[term])
                )
            )
            file.write("\n")


def load_query(query_file: str) -> list[str]:
    """Loads a query from a file and returns it as a list of lowercase strings."""
    with open(query_file, "r") as file:
        # list comprehension to read the query file and convert it to a list of lowercase strings
        query = [val.lower() for line in file for val in line.split()]
    return query


def calculate_frequencies(query):
    """Calculates the frequencies of each term in the query."""
    stemmer: RSLPStemmer = RSLPStemmer()
    query_freqs = {}
    for word in query:
        if word != "&":  # ignore the & symbol
            # stem the word and add it to the dictionary
            query_freqs[stemmer.stem(word)] = query_freqs.get(stemmer.stem(word), 0) + 1
    return query_freqs


def calculate_idfs(doc_sort, docs):
    """Calculates the IDF values of each term in the inverted index."""
    # idfs = log10(N / df)
    idfs = {term: math.log10(len(docs) / len(doc_sort[term])) for term in doc_sort}
    print("\033[1;34m==== Inverse Document Frequencies (IDFs) ====\033[0m")
    for term, value in idfs.items():
        print(f"{term}: {value}")
    return idfs


def calculate_weights(doc_sort, idfs):
    """Calculates the weights of each term in the inverted index."""
    # weights = (1 + log10(tf)) * idf
    weights = {
        (term, doc_id): (1 + math.log10(doc_sort[term][doc_id])) * idfs[term]
        for term in doc_sort
        for doc_id in doc_sort[term]
    }
    print("\n\033[1;34m==== Term weights ====\033[0m")
    for (term, doc_id), weight in weights.items():
        print(f"{term}, {doc_id}: {weight}")
    return weights


def calculate_document_weights(weights):
    """Calculates the weights of each term in each document."""
    doc_weights: dict[int, dict[str, float]] = {}
    # Reorganize the weights dictionary to make it easier to calculate the document weights
    for term, doc_id in weights:
        if doc_id not in doc_weights:
            doc_weights[doc_id] = {}
        doc_weights[doc_id][term] = weights[(term, doc_id)]

    # Print the document weights in a clear and organized format
    print("\n\033[1;34m==== Document weights ====\033[0m")
    for doc_id in sorted(doc_weights):
        print(f"Document {doc_id}:")
        for term, weight in doc_weights[doc_id].items():
            print(f"    {term}: {weight}")
    return dict(sorted(doc_weights.items(), key=lambda item: item[0]))


def calculate_query_weights(query_freqs, idfs):
    """Calculates the weights of each term in the query using the given query frequencies and IDF values."""
    query_weights: dict[str, float] = {}
    for term in query_freqs:
        if term in idfs:
            query_weights[term] = (1 + math.log10(query_freqs[term])) * idfs[term]
    print("\n\033[1;34m==== Query-term weights ====\033[0m")
    for term, weight in query_weights.items():
        print(f"{term}: {weight}")
    return query_weights


def calculate_similarities(doc_weights, query_weights, docs):
    """Calculates the similarities between the documents and the query."""
    # Similarities =  doc_term_weights / (sqrt(doc_squared_weights) * sqrt(query_squared_weights))
    similarities = {}  # similarities between the documents and the query
    doc_term_weights = {}  # document-term weights multiplied by query-term weights
    doc_squared_weights = {}  # squared sum of document-term weights
    query_squared_weights = 0  # squared sum of query-term weights

    # Initialize the dictionaries
    for doc_id in range(1, len(docs) + 1):
        similarities[doc_id] = 0
        doc_term_weights[doc_id] = 0
        doc_squared_weights[doc_id] = 0

    # Calculate the similarities
    for doc_id in range(1, len(docs) + 1):
        for term in doc_weights:
            for query_term in query_weights:
                if query_term in doc_weights[doc_id] and term == doc_id:
                    doc_term_weights[doc_id] += (
                        doc_weights[doc_id][query_term] * query_weights[query_term]
                    )

    # Calculate the squared sum of document-term weights
    for doc_id in range(1, len(docs) + 1):
        for term in doc_weights:
            if doc_id == term:
                for p in doc_weights[doc_id]:
                    doc_squared_weights[doc_id] += pow(doc_weights[doc_id][p], 2)

    # Calculate the squared sum of query-term weights
    for query_term in query_weights:
        query_squared_weights += pow(query_weights[query_term], 2)

    # Calculate the similarities
    for doc_id in range(1, len(docs) + 1):
        for term in doc_weights:
            for query_term in query_weights:
                if query_term in doc_weights[doc_id] and term == doc_id:
                    similarities[doc_id] = (doc_term_weights[doc_id]) / (
                        math.sqrt(doc_squared_weights[doc_id])
                        * math.sqrt(query_squared_weights)
                    )
    
    # Printing the results
    print(
        "\n\033[1;34m==== Document-term weights multiplied by query-term weights ====\033[0m"
    )
    for doc_id in range(1, len(docs) + 1):
        print(f"Doc {doc_id}: {doc_term_weights[doc_id]}")

    print("\n\033[1;34m==== Squared sum of document-term weights ====\033[0m")
    for doc_id in range(1, len(docs) + 1):
        print(f"Doc {doc_id}: {doc_squared_weights[doc_id]}")

    print("\n\033[1;34m==== Squared sum of query-term weights ====\033[0m")
    print(query_squared_weights)

    print("\n\033[1;34m==== Similarities ====\033[0m")
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1]))
    for doc_id in similarities:
        print(f"Doc {doc_id}: {similarities[doc_id]}")

    return similarities, doc_term_weights, doc_squared_weights, query_squared_weights


def write_weights_to_file(doc_weights, docs):
    """Writes the weights of each term in each document to a file."""
    with open("pesos.txt", "w", encoding="utf8") as file:
        for doc_id in range(1, len(docs) + 1):
            file.write(f"{docs[doc_id]}: ")
            for term, weight in doc_weights[doc_id].items():
                if weight > 0:
                    file.write(f"{term}, {weight}   ")
            file.write("\n")


def write_response_to_file(sim_scores, docs):
    """Writes the response to a file."""
    num_docs = sum(1 for doc_id in sim_scores if sim_scores[doc_id] >= 0.001)

    with open("resposta.txt", "w", encoding="utf8") as file:
        file.write(f"{num_docs}\n")
        print("\n\033[1;34m==== Response ====\033[0m")
        print(f"{num_docs}")
        for doc_id in reversed(sim_scores):
            if sim_scores[doc_id] >= 0.001:
                file.write(f"{docs[doc_id]} {sim_scores[doc_id]}\n")
                print(f"{docs[doc_id]} {sim_scores[doc_id]}")


def vector_space_model(doc_sort, docs, query_file):
    """Combines the functions above to implement the vector space model."""
    query = load_query(query_file)
    query_freqs = calculate_frequencies(query)
    idfs = calculate_idfs(doc_sort, docs)
    weights = calculate_weights(doc_sort, idfs)
    doc_weights = calculate_document_weights(weights)
    query_weights = calculate_query_weights(query_freqs, idfs)
    (
        similarities,
        doc_term_weights,
        doc_squared_weights,
        query_squared_weights,
    ) = calculate_similarities(doc_weights, query_weights, docs)
    write_weights_to_file(doc_weights, docs)
    write_response_to_file(similarities, docs)


def measure_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator function to measure the execution time of a function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        end_time: float = time.time()
        total_time: float = end_time - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        return result

    return wrapper


@measure_execution_time
def main(base_filename: str, query_filename: str) -> None:
    """Main function."""
    initialize_nltk()  # Initialize NLTK and download the necessary resources

    # --- Load the documents ---
    print("\033[1;37m📚 Loading documents...\033[0m")
    time.sleep(0.75) # if you want to run the program faster, you can comment this line
    texts, documents = load_documents(base_filename)
    print(f"\033[1;32;40m✅ Loaded {len(documents)} documents!\033[0m")

    # --- Build the inverted index from the documents ---
    print("\033[1;36m🔍 Building inverted index...\n\033[0m")
    time.sleep(0.75)

    # --- Declare the variables for the inverted index ---
    tagger = load_tagger()
    stemmer = RSLPStemmer()

    # --- Build the inverted index and save it to a file ---
    document_sort = build_inverted_index(texts, tagger, stemmer)
    save_inverted_index(document_sort)
    print(f"\033[1;32;40m✅ Built inverted index!\033[0m")

    # --- Load the query ---
    print("\033[1;34m📝 Loading query...\033[0m")
    time.sleep(0.75)
    query = query_filename
    # query = load_query(query_filename)

    # --- Evaluate the query ---
    print("\033[1;35m🔎 Evaluating query...\033[0m")
    time.sleep(0.75)

    # Display the inverted index in the terminal
    print("\n\033[1;34m==== Inverted Index ====\033[0m")
    display_inverted_index(document_sort)
    print()

    # --- Calls the method that handles the vector space model ---
    vector_space_model(document_sort, documents, query)

    # Display the execution information in the terminal
    print("\n\033[1;34m==== Execution Information ====\033[0m")
    print(f"Base File: {base_filename}")
    print(f"Query File: {query_filename}")

    print("\n\033[1;33m✨ Done!\033[0m\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "\033[91m\n❌ ERROR: Invalid arguments! Please use the following format:\n"
        )
        print("\033[93mUsage: python nome_do_programa.py base.txt consulta.txt\n")
        print(
            "\033[96mAlternatively, you can run the task script by using VSCode's shortcut Ctrl+Shift+B.\n"
        )
        print(
            "\033[92mTip: you can use the command './delete.bat' (on Windows) or './delete.sh' (on Linux) to delete the output files,"
        )
        print("generating new output files instead of overwriting the old ones.")
        print("\033[0m")
        sys.exit(1)
    main(base_filename=sys.argv[1], query_filename=sys.argv[2])
