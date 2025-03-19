import json
import os
import pprint

from datasets import load_dataset
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from tqdm import tqdm

# Load API keys from .env file
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "llama3.3-70b-instruct")

# Ensure API key is set
if not API_KEY:
    raise ValueError("Missing API_KEY in .env file.")

# Initialize Llama API Client
client = LlamaStackClient(base_url=API_BASE_URL, api_key=API_KEY)

# List available models (debugging step)
available_models = client.models.list()
pprint.pprint(available_models)


# Load LongBench v2 dataset (train split)
def load_longbench():
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    print("Loaded dataset successfully.")
    print("Example sample:", dataset[0])  # Debug: Check structure
    return dataset


# Query Llama API using the LLaMA model
def query_llama(question, context, choices):
    # Format the prompt properly with multiple-choice options
    prompt = f"""Context:\n{context}\n\n
Question: {question}
A) {choices['A']}
B) {choices['B']}
C) {choices['C']}
D) {choices['D']}

Which option is correct? Answer only with 'A', 'B', 'C', or 'D'."""

    try:
        response = client.inference.chat_completion(
            messages=[UserMessage(role="user", content=prompt)],
            model_id=MODEL_ID,
            stream=False,
        )

        # Debugging: Print the response structure
        print(f"🔍 Raw Response: {response}")

        # Try extracting content correctly
        return response.message.content.strip()

    except Exception as e:
        print(f"❌ Llama API Error: {e}")
        return None


# Evaluate model performance
def evaluate():
    dataset = load_longbench()
    results = []

    for sample in tqdm(dataset, desc="Evaluating"):
        question = sample.get("question", "")
        context = sample.get("context", "")
        correct_answer = sample.get("answer", "")

        choices = {
            "A": sample.get("choice_A", ""),
            "B": sample.get("choice_B", ""),
            "C": sample.get("choice_C", ""),
            "D": sample.get("choice_D", ""),
        }

        if not question or not context or not correct_answer:
            continue  # Skip invalid samples

        # Get model prediction
        model_output = query_llama(question, context, choices)

        # Compare with correct answer
        is_correct = model_output == correct_answer

        results.append(
            {
                "question": question,
                "expected": correct_answer,
                "model_output": model_output,
                "correct": is_correct,
            }
        )

    # Prevent division by zero
    if not results:
        print("No valid results collected. Check dataset structure or API response.")
        return

    # Calculate accuracy
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"Accuracy: {accuracy:.2%}")

    # Save results to JSON file
    with open("longbench_eval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Results saved to longbench_eval_results.json")


if __name__ == "__main__":
    evaluate()
