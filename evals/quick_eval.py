import os

from dotenv import load_dotenv
from groq import Groq
from config import getenv

GROQ_API_KEY = getenv("GROQ_API_KEY")


def simple_evaluator(response, expected):
    """A simple evaluator that scores based on exact match and length similarity."""
    if response.strip().lower() == expected.strip().lower():
        return 1.0
    return max(
        0, 1 - abs(len(response) - len(expected)) / max(len(response), len(expected), 1)
    )


def query_groq_api(prompt):
    """Function to query the Groq API with LLaMA 3 8B model using the Groq client."""
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
    )
    return (
        response.choices[0].message.content
        if response.choices
        else "Error: API request failed"
    )


def run_eval():
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
        response = query_groq_api(scenario["question"])

        # Evaluate response
        score = simple_evaluator(response, scenario["expected"])

        eval_results.append(
            {
                "model": "llama3-8b-8192",
                "question": scenario["question"],
                "response": response,
                "expected": scenario["expected"],
                "score": score,
            }
        )

    # Display results
    for result in eval_results:
        print(f"Model: {result['model']}")
        print(f"Question: {result['question']}")
        print(f"Response: {result['response']}")
        print(f"Expected: {result['expected']}")
        print(f"Score: {result['score']:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    run_eval()
