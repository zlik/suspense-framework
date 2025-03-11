import numpy as np
import requests
import together
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
API_KEY = "your_api_key_here"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EMBEDDING_MODEL = "togethercomputer/m2-bert-80M-32k-retrieval"

# Initialize Together AI client
client = together.Client(api_key=API_KEY)

# List of books to download from Project Gutenberg
books = {
    "Pride and Prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "Moby-Dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    # "Ulysses": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
    # "Don Quixote": "https://www.gutenberg.org/cache/epub/996/pg996.txt",
    # "Great Expectations": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    # "The Odyssey": "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
    # "The Count of Monte Cristo": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
}


def download_book(url):
    """Downloads book text from Project Gutenberg and removes metadata."""
    response = requests.get(url)
    book_text = response.text

    start_idx = book_text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end_idx = book_text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    if start_idx != -1 and end_idx != -1:
        return book_text[start_idx:end_idx].strip()
    return ""


def count_tokens_together(text):
    """Counts the number of tokens in a given text using Together AI's API."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
        temperature=0,
    )
    return response.usage.prompt_tokens if response.usage else "Token count unavailable"


def get_embedding(text):
    """Gets an embedding vector for the given text using Together AI's embedding model."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def evaluate_model_responses(book_title, passage):
    """Evaluates the model's ability to handle large-context passages."""
    evaluation_metrics = {}

    queries = {
        "Summary Coherence": f"Summarize this passage: {passage}",
        "Theme Extraction": "What key themes are present in this passage?",
        "Consistency Check": "Can you provide a second summary of the passage?",
    }

    responses = {}
    for key, query in queries.items():
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": query}],
            temperature=0,
        )
        responses[key] = response.choices[0].message.content

    # Get embeddings for consistency check
    summary_1_embedding = get_embedding(responses["Summary Coherence"])
    summary_2_embedding = get_embedding(responses["Consistency Check"])

    # Compute cosine similarity for consistency
    consistency_score = cosine_similarity([summary_1_embedding], [summary_2_embedding])[
        0
    ][0]

    # Evaluation Scores
    evaluation_metrics["Summary Length"] = len(responses["Summary Coherence"].split())
    evaluation_metrics["Theme Count"] = len(responses["Theme Extraction"].split(","))
    evaluation_metrics["Consistency Score"] = (
        consistency_score  # Based on actual similarity
    )

    return evaluation_metrics


def process_text_chunks(clean_text, token_sizes):
    """Processes text chunks of different sizes, collects token counts, and evaluates them."""
    results = []
    words = clean_text.split()

    for size in token_sizes:
        chunk_text = " ".join(words[:size])
        token_count = count_tokens_together(chunk_text)
        evaluation = evaluate_model_responses("Unknown Book", chunk_text)
        results.append(
            {"Chunk Size (Tokens)": size, "Token Count": token_count, **evaluation}
        )

    return results


# Process each book and compare large text chunks
token_sizes = [1000, 10000, 25000, 50000, 75000, 100000, 125000]
chunk_comparisons = []

for title, url in books.items():
    print(f"Processing: {title}")
    clean_text = download_book(url)

    # Process different chunk sizes
    chunk_results = process_text_chunks(clean_text, token_sizes)

    for res in chunk_results:
        res["Title"] = title
        chunk_comparisons.append(res)

# Print chunk comparison results
for res in chunk_comparisons:
    print("\nBook: {Title}".format(**res))
    print("Chunk Size (Tokens): {Chunk Size (Tokens)}".format(**res))
    print("Token Count: {Token Count}".format(**res))
    print("Summary Length: {Summary Length}".format(**res))
    print("Theme Count: {Theme Count}".format(**res))
    print("Consistency Score: {Consistency Score}".format(**res))
