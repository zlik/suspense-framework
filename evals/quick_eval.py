import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from config import getenv
from llama_stack_client import LlamaStackClient
from tabulate import tabulate
from colorama import Fore, Style
from questions.evaluation_questions import evaluation_scenarios  # Import questions

# Load API credentials
LLAMA_API_KEY = getenv("LLAMA_API_KEY")
LLAMA_API_BASE_URL = getenv("LLAMA_API_BASE_URL")

# Initialize Llama client
llama_client = LlamaStackClient(base_url=LLAMA_API_BASE_URL, api_key=LLAMA_API_KEY)

# Define different context window lengths to test
context_sizes = [128, 256]  # Varying context lengths

def simple_evaluator(response, expected):
    """A simple evaluator that scores based on exact match and length similarity."""
    if response.strip().lower() == expected.strip().lower():
        return 1.0
    return max(0, 1 - abs(len(response) - len(expected)) / max(len(response), len(expected), 1))

def query_llama_api(prompt, model="llama3.3-70b-llama_api", max_tokens=2048):
    """Function to query the Llama API with a specified model and context length."""
    truncated_prompt = prompt[:max_tokens]  # Truncate prompt to fit context window
    try:
        response = llama_client.inference.chat_completion(
            model_id=model,
            messages=[{"role": "user", "content": truncated_prompt}],
        )
        return response.completion_message.content.text if response else "Error: API request failed"
    except Exception as e:
        return f"Error: {str(e)}"

def run_eval(models, context_sizes):
    eval_results = []
    tasks = []

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        for scenario in evaluation_scenarios:
            for context_size in context_sizes:
                for model in models:
                    tasks.append(
                        executor.submit(
                            query_llama_api, scenario["question"], model, context_size
                        )
                    )

        # Collect results as they complete
        for future, (scenario, context_size, model) in zip(as_completed(tasks), [
            (s, cs, m) for s in evaluation_scenarios for cs in context_sizes for m in models
        ]):
            response = future.result()
            score = simple_evaluator(response, scenario["expected"])
            eval_results.append({
                "question": scenario["question"],
                "expected": scenario["expected"],
                "context_size": context_size,
                "model": model,
                "response": response[:50] + "...",
                "score": score,
            })

    # Display results
    headers = ["Question", "Expected", "Context Size", "Model", "Response", "Score"]
    table_data = [
        [
            Fore.CYAN + result["question"][:25] + "..." + Style.RESET_ALL,
            Fore.GREEN + result["expected"][:25] + "..." + Style.RESET_ALL,
            Fore.MAGENTA + str(result["context_size"]) + Style.RESET_ALL,
            Fore.BLUE + result["model"] + Style.RESET_ALL,
            Fore.YELLOW + result["response"] + Style.RESET_ALL,
            f"{result['score']:.2f}",
        ]
        for result in eval_results
    ]

    print(tabulate(table_data, headers=headers, tablefmt="plain"))

if __name__ == "__main__":
    models_to_compare = {"llama3.3-8b-llama_api", "llama3.3-70b-llama_api"}
    run_eval(models=models_to_compare, context_sizes=context_sizes)
