import os
from dotenv import load_dotenv
from groq import Groq
from config import getenv
from llama_stack_client import LlamaStackClient
from tabulate import tabulate
from colorama import Fore, Style

GROQ_API_KEY = getenv("GROQ_API_KEY")
LLAMA_API_KEY = getenv("LLAMA_API_KEY")
LLAMA_API_BASE_URL = getenv("LLAMA_API_BASE_URL")

# Initialize Llama client
llama_client = LlamaStackClient(base_url=LLAMA_API_BASE_URL, api_key=LLAMA_API_KEY)


def simple_evaluator(response, expected):
    """A simple evaluator that scores based on exact match and length similarity."""
    if response.strip().lower() == expected.strip().lower():
        return 1.0
    return max(
        0, 1 - abs(len(response) - len(expected)) / max(len(response), len(expected), 1)
    )


def query_llama_api(prompt, model="llama3.3-70b-llama_api"):
    """Function to query the Llama API with a specified model."""
    response = llama_client.inference.chat_completion(
        model_id=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.completion_message.content.text if response else "Error: API request failed"


def query_model_api(prompt, model=None):
    """Unified function to query Llama API with different models."""
    return query_llama_api(prompt, model=model)


def run_eval(models):
    # Define evaluation scenarios
    scenarios = [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {
            "question": "Summarize the main idea of the Theory of Relativity.",
            "expected": "The theory states that space and time are relative and interwoven, with gravity affecting time and space curvature.",
        },
    ]

    # Initialize evaluation results
    eval_results = []

    for scenario in scenarios:
        model_responses = {}
        for model in models:
            response = query_model_api(scenario["question"], model=model)
            score = simple_evaluator(response, scenario["expected"])
            model_responses[model] = {"response": response[:50] + "...", "score": score}  # Truncate response

        eval_results.append(
            {
                "question": scenario["question"],
                "expected": scenario["expected"],
                "models": model_responses,
            }
        )

    # Display results in a compact tabulated format
    table_data = []
    headers = ["Question", "Expected"] + list(models)
    for result in eval_results:
        row = [
            Fore.CYAN + result["question"][:25] + "..." + Style.RESET_ALL,  # Shorten questions
            Fore.GREEN + result["expected"][:25] + "..." + Style.RESET_ALL,  # Shorten expected answers
        ]
        for model in models:
            response = result["models"].get(model, {}).get("response", "N/A")
            score = result["models"].get(model, {}).get("score", 0)
            row.append(Fore.YELLOW + response + Style.RESET_ALL + f" ({score:.2f})")
        table_data.append(row)

    print(tabulate(table_data, headers =headers, tablefmt="plain"))  # Use 'plain' format for compact output

if __name__ == "__main__":
    models_to_compare = {
        "llama3.3-8b-llama_api",
        "llama3.3-70b-llama_api",
    }
    run_eval(models=models_to_compare)
