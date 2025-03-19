import numpy as np
import requests
import together

# Configuration
API_KEY = "YOUR_LLAMA_API_KEY"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Initialize Together AI client
client = together.Client(api_key=API_KEY)

# List of books to download from Project Gutenberg (excluding Russian authors)
books = {
    "Pride and Prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "Moby-Dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
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


def quick_evaluate(text):
    """Performs a quick evaluation by summarizing the text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Summarize this passage: {text}"}],
        temperature=0,
    )
    summary = response.choices[0].message.content
    return len(summary.split())


def process_text_chunks(clean_text, token_sizes):
    """Processes text chunks quickly by only counting tokens and generating a short summary."""
    results = []
    words = clean_text.split()

    for size in token_sizes:
        chunk_text = " ".join(words[:size])
        token_count = count_tokens_together(chunk_text)
        summary_length = quick_evaluate(chunk_text)
        results.append(
            {
                "Chunk Size (Tokens)": size,
                "Token Count": token_count,
                "Summary Length": summary_length,
            }
        )

    return results


# Process each book and compare large text chunks
token_sizes = [1000, 10000, 25000]
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
