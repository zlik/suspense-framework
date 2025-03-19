import opik


def run_eval():
    # Define evaluation scenarios
    scenarios = [
        {"question": "What is the capital of France?", "expected": "Paris"},
        {
            "question": "Summarize the main idea of the Theory of Relativity.",
            "expected": "The theory states that space and time are relative and interwoven, with gravity affecting time and space curvature.",
        },
    ]

    # Define model settings with different context windows
    model_configs = [
        {"name": "model_1k", "context_window": 1000},
        {"name": "model_2k", "context_window": 2000},
    ]

    # Initialize Opik evaluation
    eval_results = []

    for config in model_configs:
        for scenario in scenarios:
            response = opik.run_model(
                model_name=config["name"],
                prompt=scenario["question"],
                max_tokens=config["context_window"],
            )
            eval_results.append(
                {
                    "model": config["name"],
                    "question": scenario["question"],
                    "response": response,
                    "expected": scenario["expected"],
                    "score": opik.evaluate(response, scenario["expected"]),
                }
            )

    # Display results
    for result in eval_results:
        print(f"Model: {result['model']}")
        print(f"Question: {result['question']}")
        print(f"Response: {result['response']}")
        print(f"Expected: {result['expected']}")
        print(f"Score: {result['score']}")
        print("-" * 50)


if __name__ == "__main__":
    run_eval()
