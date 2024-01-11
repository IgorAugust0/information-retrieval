#  Autor: Igor Augusto Reis Gomes   [12011BSI290]

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
from importlib import import_module


def install_package(package_name: str):
    """Function to install a package using pip. If the package is already installed, it will be skipped."""
    try:
        import_module(package_name)
        print(f"\033[1;32;40m✅ {package_name} is installed!\033[0m")
    except ImportError:
        print(f"\033[1;31m⚠ {package_name} not found. Installing it...\033[0m")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            import_module(package_name)  # Try to import again after installation
            print(f"\033[1;32;40m✅ {package_name} is now installed!\033[0m")
        except Exception as e:
            print(f"\033[1;31m⚠ Failed to install {package_name}: {e}\033[0m")
            if package_name == "nltk":
                sys.exit(1)


def download_nltk_resources(resources: list[str]):
    """Download the necessary resources from NLTK."""
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[1])


def initialize_packages():
    """Initialize packages and download the necessary resources."""
    packages = ["nltk", "prettytable"]
    for package in packages:
        install_package(package)
    
    if "nltk" in packages:
        resources = ["tokenizers/punkt", "corpora/stopwords", "stemmers/rslp"]
        download_nltk_resources(resources)


def load_tagger() -> UnigramTagger:
    """
    Load or create the NLTK tagger and save it to a binary file.

    Returns:
        UnigramTagger: The NLTK UnigramTagger.
    """
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
    """
    Load document filenames and contents from a base file.

    Args:
        base_filename (str): The filename of the base file.

    Returns:
        tuple[list[str], dict[int, str]]: A tuple containing a list of document texts and a dictionary mapping
        document IDs to their filenames.
    """
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
    """
    Preprocess the given text.

    Args:
        text (str): The text to preprocess.
        tagger (UnigramTagger): The NLTK UnigramTagger for part-of-speech tagging.
        stemmer (RSLPStemmer): The NLTK RSLPStemmer for stemming.
        show_classifications (bool, optional): Whether to display word classifications in the terminal. Defaults to False.

    Returns:
        dict[str, int]: A dictionary mapping stemmed words to their frequencies.
    """
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
        print("\n\033[1;35m==== Complete Word Classification ====\033[0m")
        print_classifications(global_classification_list)
        # for classification in global_classification_list:
        #     print(f"word - classification: {classification}")
        print()

    return word_count


def print_classifications(global_classification_list):
    """Print the word-classification pairs in the given list."""
    for pair in global_classification_list:
        print(f"word - classification: {pair}")


def build_inverted_index(
    texts: list[str], tagger: UnigramTagger, stemmer: RSLPStemmer
) -> dict[str, dict[int, int]]:
    """
    Build an inverted index from the preprocessed documents.

    Args:
        texts (list[str]): A list of preprocessed document texts.
        tagger (UnigramTagger): The NLTK UnigramTagger for part-of-speech tagging.
        stemmer (RSLPStemmer): The NLTK RSLPStemmer for stemming.

    Returns:
        dict[str, dict[int, int]]: The inverted index mapping terms to document IDs and term frequencies.
    """
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
    """
    Display the inverted index in the terminal.

    Args:
        inverted_index (dict[str, dict[int, int]]): The inverted index mapping terms to document IDs and term frequencies.

    Returns:
        None
    """
    try:
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = ["Term", "Documents"]
        for term in inverted_index:
            docs = ""
            for doc_id in inverted_index[term]:
                docs += f"{doc_id},{inverted_index[term][doc_id]} "
            table.add_row([term, docs])
        print(table)
    except ImportError:
        print(
            "\033[1;31m⚠ prettytable not found. Displaying inverted index without it.\033[0m"
        )
        for term in inverted_index:
            print(f"{term}: ", end="")
            for doc_id in inverted_index[term]:
                print(f"{doc_id},{inverted_index[term][doc_id]} ", end="")
            print()
        return

    # without using PrettyTable
    # for term in inverted_index:
    #     print(f"{term}: ", end="")
    #     for doc_id in inverted_index[term]:
    #         print(f"{doc_id},{inverted_index[term][doc_id]} ", end="")
    #     print()


def save_inverted_index(inverted_index: dict[str, dict[int, int]]) -> None:
    """
    Save the inverted index to a file.

    Args:
        inverted_index (dict[str, dict[int, int]]): The inverted index mapping terms to document IDs and term frequencies.

    Returns:
        None
    """
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
    """
    Load the query from the given file.

    Args:
        query_file (str): The filename of the query file.

    Returns:
        list[str]: A list of query terms.
    """
    with open(query_file, "r") as file:
        query: list[str] = file.readline().split()

    for i, val in enumerate(query):
        query[i] = val.lower()

    return query


def evaluate_query(
    query: str, document_sort: dict[str, dict[int, int]], documents: dict[int, str]
) -> dict[int, list[bool]]:
    """
    Evaluate the given query using a boolean model.

    Args:
        query (str): The query string.
        document_sort (dict[str, dict[int, int]]): The inverted index mapping terms to document IDs and term frequencies.
        documents (dict[int, str]): A dictionary mapping document IDs to their filenames.

    Returns:
        dict[int, list[bool]]: A dictionary mapping document IDs to a list of Boolean results for the query.
    """
    query_terms: list[str] = load_query(query)

    def search_word(word: str, doc_id: int) -> bool:
        """Search for the given word in the given document."""
        stemmer: RSLPStemmer = RSLPStemmer()
        for term, doc_freqs in document_sort.items():
            if stemmer.stem(word) == term:
                for doc, _ in doc_freqs.items():
                    if doc == doc_id:
                        return True
        return False

    query_results: dict[int, list[bool]] = {}

    single_term: bool = True
    and_operator_count: int = 0

    # Processing query terms
    for i, term in enumerate(query_terms):
        if term == "&":
            and_operator_count += 1

            # Evaluating conjunctions (AND)
            for doc_id in range(1, len(documents) + 1):
                if query_terms[i - 1][0] == "!":
                    term1_result: bool = not search_word(query_terms[i - 1][1:], doc_id)
                else:
                    term1_result: bool = search_word(query_terms[i - 1], doc_id)

                if query_terms[i + 1][0] == "!":
                    term2_result: bool = not search_word(query_terms[i + 1][1:], doc_id)
                else:
                    term2_result: bool = search_word(query_terms[i + 1], doc_id)

                term_final_result: bool = term1_result and term2_result

                if doc_id in query_results:
                    query_results[doc_id].append(term_final_result)
                else:
                    query_results[doc_id] = [term_final_result]

            single_term = False

    # Processing single query terms
    for i, term in enumerate(query_terms):
        if single_term:
            for doc_id in range(1, len(documents) + 1):
                if term[0] == "!":
                    term_result: bool = not search_word(term[1:], doc_id)
                else:
                    term_result: bool = search_word(term, doc_id)

                query_results.setdefault(doc_id, []).append(term_result)

    # Processing disjunctions (OR) and more complex queries
    for i, term in enumerate(query_terms):
        for doc_id in range(1, len(documents) + 1):
            if term == "|":
                if len(query_results) != 0:
                    if query_terms[i - 2] != "&":
                        if query_terms[i - 1][0] == "!":
                            term1_result: bool = not search_word(
                                query_terms[i - 1][1:], doc_id
                            )
                        else:
                            term1_result: bool = search_word(query_terms[i - 1], doc_id)

                        term2_result: bool = query_results[doc_id][-1]

                        term_final_result: bool = term1_result or term2_result

                        query_results.setdefault(doc_id, []).append(term_final_result)

                    if len(query_terms) - 2 == i:
                        if len(query_terms) == 7:
                            term1_result: bool = query_results[doc_id][-2]
                        else:
                            term1_result: bool = query_results[doc_id][-1]

                        if query_terms[i + 1][0] == "!":
                            term2_result: bool = not search_word(
                                query_terms[i + 1][1:], doc_id
                            )
                        else:
                            term2_result: bool = search_word(query_terms[i + 1], doc_id)

                        term_final_result: bool = term1_result or term2_result

                        query_results.setdefault(doc_id, []).append(term_final_result)

    # Final processing for complex queries
    for doc_id in range(1, len(documents) + 1):
        if len(query_terms) > 3:
            term1_result: bool = query_results[doc_id][-1]
            term2_result: bool = query_results[doc_id][-2]

            term_final_result: bool = (
                term1_result and term2_result
                if and_operator_count > 1
                else term1_result or term2_result
            )

            query_results.setdefault(doc_id, []).append(term_final_result)

    return query_results  # Return the results


def save_results(
    query_results: dict[int, list[bool]], documents: dict[int, str]
) -> None:
    """
    Save query results to a file and display them.

    Args:
        query_results (dict[int, list[bool]]): A dictionary mapping document IDs to a list of Boolean results for the query.
        documents (dict[int, str]): A dictionary mapping document IDs to their filenames.

    Returns:
        None
    """
    count: int = sum(1 for doc_id in query_results if query_results[doc_id][-1])

    if count > 0:
        with open("resposta.txt", encoding="utf8", mode="w") as file:
            print(count, file=file)
            for doc_id in query_results:
                if query_results[doc_id][-1]:
                    file.write(documents[doc_id] + "\n")
    else:
        print("No results found.")


def display_query_results(
    query_results: dict[int, list[bool]],
    documents: dict[int, str],
    query_file_path: str,
) -> None:
    """
    Display the query results in the terminal.

    Args:
        query_results (dict[int, list[bool]]): A dictionary mapping document IDs to a list of Boolean results for the query.
        documents (dict[int, str]): A dictionary mapping document IDs to their filenames.
        query_file_path (str): The path to the query file.

    Returns:
        None
    """
    count: int = sum(1 for doc_id in query_results if query_results[doc_id][-1])
    num_docs: int = len(query_results)

    if count > 0:
        with open(query_file_path, encoding="utf8", mode="r") as query_file:
            query = query_file.read().strip()
        print(f"\nQuery: {query}")
        print(f"{count} out of {num_docs} documents matched the query.")
        print(f"Check the file 'resposta.txt' for the results.\n")
        try:
            from prettytable import PrettyTable

            table = PrettyTable()
            table.field_names = ["Document Name", "Result"]
            for doc_id in query_results:
                if query_results[doc_id][-1]:
                    table.add_row([documents[doc_id], "Yes"])
                else:
                    table.add_row([documents[doc_id], "No"])
            print(table)
        except ImportError:
            print(
                "\033[1;31m⚠ prettytable not found. Displaying query results without it.\033[0m"
            )
            for doc_id in query_results:
                if query_results[doc_id][-1]:
                    print(f"{documents[doc_id]}: Yes")
                else:
                    print(f"{documents[doc_id]}: No")
            return


def display_detailed_results(query_results: dict[int, list[bool]]) -> None:
    """
    Display the detailed query results in a table.

    Args:
        query_results (dict[int, list[bool]]): A dictionary mapping document IDs to a list of Boolean results for the query.

    Returns:
        None
    """
    try:
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = ["Document", "Result"]
        for doc_id, result in query_results.items():
            table.add_row([doc_id, result[-1]])
        print(table)
    except ImportError:
        print(
            "\033[1;31m⚠ prettytable not found. Displaying detailed results without it.\033[0m"
        )
        for doc_id, result in query_results.items():
            print(f"Document {doc_id}: {result}")
        return


def display_output(
    document_sort: dict[str, dict[int, int]],
    query_results: dict[int, list[bool]],
    documents: dict[int, str],
    base_filename: str,
    query_filename: str,
) -> None:
    """
    Display the output in the terminal.

    Args:
        document_sort (dict[str, dict[int, int]]): The inverted index mapping terms to document IDs and term frequencies.
        query_results (dict[int, list[bool]]): A dictionary mapping document IDs to a list of Boolean results for the query.
        documents (dict[int, str]): A dictionary mapping document IDs to their filenames.
        base_filename (str): The filename of the base file.
        query_filename (str): The filename of the query file.

    Returns:
        None
    """
    # Display the word classifications in the terminal
    # To display the complete list of word classifications, set show_classifications to True in preprocess_text(), line 199
    # print("\n\033[1;34m==== Word Classification ====\033[0m")
    # print_classifications(global_classification_list)
    print()

    # Display the inverted index in the terminal
    print("\033[1;34m==== Inverted Index ====\033[0m")
    display_inverted_index(document_sort)
    print()

    # Display the query results in the terminal
    print("\033[1;34m==== Query Results ====\033[0m")
    display_query_results(query_results, documents, query_file_path=query_filename)
    print()

    # Display the detailed query results in the terminal
    print("\033[1;34m==== Detailed Results ====\033[0m")
    display_detailed_results(query_results)
    print()

    # Display the execution information in the terminal
    print("\033[1;34m==== Execution Information ====\033[0m")
    print(f"Base File: {base_filename}")
    print(f"Query File: {query_filename}")


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
def main(base_filename: str, query_filename: str) -> None:
    """
    Main function.

    Args:
        base_filename (str): The filename of the base file.
        query_filename (str): The filename of the query file.

    Returns:
        None
    """
    initialize_packages() # Initialize packages and download the necessary resources
    # --- Load the documents ---
    print("\033[1;37m📚 Loading documents...\033[0m")
    time.sleep(1)
    texts, documents = load_documents(base_filename)
    print(f"\033[1;32;40m✅ Loaded {len(documents)} documents!\033[0m")

    # --- Build the inverted index from the documents ---
    print("\033[1;36m🔍 Building inverted index...\n\033[0m")
    time.sleep(1)

    # --- Declare the variables for the inverted index ---
    tagger = load_tagger()
    stemmer = RSLPStemmer()

    # --- Build the inverted index and save it to a file ---
    document_sort = build_inverted_index(texts, tagger, stemmer)
    save_inverted_index(document_sort)
    print(f"\033[1;32;40m✅ Built inverted index!\033[0m")

    # --- Load the query ---
    print("\033[1;34m📝 Loading query...\033[0m")
    time.sleep(1)
    query = query_filename
    # query = load_query(query_filename)

    # --- Evaluate the query ---
    print("\033[1;35m🔎 Evaluating query...\033[0m")
    time.sleep(1)
    results_all = {}

    # --- Evaluate the query and save the results to a file ---
    try:
        from tqdm import trange

        total_iterations = int(len(documents) / 2)
        for _ in trange(
            total_iterations,
            desc="\033[1;31m🚀 Query progress\033[0m",
            bar_format="{l_bar}\033[1;31m{bar:50}\033[0m{r_bar}",
        ):
            query_results = evaluate_query(query, document_sort, documents)
            results_all.update(query_results)
            save_results(results_all, documents)
            time.sleep(0.1)
    except ImportError:
        print("\033[1;31m⚠ tqdm not found. Running query without progress bar.\033[0m")
        for _, _ in enumerate(documents):
            query_results = evaluate_query(query, document_sort, documents)
            results_all.update(query_results)
            save_results(results_all, documents)
        print(f"\033[1;32;40m✅ Evaluated query!\033[0m")

    print("\033[1;33m✨ Done!\033[0m")

    # --- Display the output ---
    display_output(
        document_sort,
        results_all,
        documents,
        base_filename,
        query_filename,
    )


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
    # base_filename, query_filename = sys.argv[1], sys.argv[2]
    main(base_filename=sys.argv[1], query_filename=sys.argv[2])
