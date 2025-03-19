import requests
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage

# Configuration
MODEL = "llama3.3-70b-instruct"

client = LlamaStackClient(
    base_url="https://api.llama.com/",
    api_key="LLAMA_API_KEY",
)


# List of books to download from Project Gutenberg
books = {
    "Pride and Prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "Moby-Dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
}


def download_book(url):
    """Downloads book text from Project Gutenberg and removes metadata."""
    response = requests.get(url)
    book_text = response.text

    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

    start_idx = book_text.find(start_marker)
    end_idx = book_text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        return book_text[start_idx + len(start_marker) : end_idx].strip()
    return book_text.strip()


def query_model(query):
    """Queries the Llama model and retrieves a response."""
    try:
        response = client.inference.chat_completion(
            model_id=MODEL, messages=[UserMessage(role="user", content=query)]
        )

        # Extract response content correctly
        content = response.completion_message.content
        if isinstance(content, list):  # Handle case where content is a list
            content = " ".join(item.text for item in content if hasattr(item, "text"))
        elif hasattr(content, "text"):  # Handle single TextContentItem case
            content = content.text

        return content if content else "No response"

    except Exception as e:
        return f"Error: {str(e)}"


def evaluate_model_responses(passage):
    """Evaluates the model's ability to handle large-context passages."""
    evaluation_metrics = {}

    queries = {
        "Summary Coherence": f"Summarize this passage: {passage}",
        "Theme Extraction": f"What key themes are present in this passage?",
        "Consistency Check": f"Provide a second independent summary of this passage.",
    }

    responses = {key: query_model(query) for key, query in queries.items()}

    # Ensure responses are strings before calling .split()
    evaluation_metrics["Summary Length"] = (
        len(responses["Summary Coherence"].split())
        if isinstance(responses["Summary Coherence"], str)
        else 0
    )
    evaluation_metrics["Theme Count"] = (
        len(responses["Theme Extraction"].split(","))
        if isinstance(responses["Theme Extraction"], str)
        else 0
    )
    evaluation_metrics["Consistency Check"] = responses.get("Consistency Check", "N/A")

    return evaluation_metrics


def process_text_chunks(clean_text, token_sizes):
    """Processes text chunks of different sizes and evaluates them."""
    results = []
    words = clean_text.split()

    for size in token_sizes:
        chunk_text = " ".join(words[:size])
        evaluation = evaluate_model_responses(chunk_text)
        results.append({"Chunk Size (Tokens)": size, **evaluation})

    return results


# Process each book and compare large text chunks
token_sizes = [1000, 10000, 25000]
chunk_comparisons = []

for title, url in books.items():
    print(f"\nProcessing: {title}")
    clean_text = download_book(url)

    # Process different chunk sizes
    chunk_results = process_text_chunks(clean_text, token_sizes)

    for res in chunk_results:
        res["Title"] = title
        chunk_comparisons.append(res)

# Print chunk comparison results
for res in chunk_comparisons:
    print(f"\nBook: {res['Title']}")
    print(f"Chunk Size (Tokens): {res['Chunk Size (Tokens)']}")
    print(f"Summary Length: {res['Summary Length']}")
    print(f"Theme Count: {res['Theme Count']}")
    print(f"Consistency Check: {res['Consistency Check']}")
