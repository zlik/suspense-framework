from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets.evaluation_questions import evaluation_scenarios
from evaluator import simple_evaluator
from inference.llama_api_client import query_llama_api

def run_eval(models, context_sizes):
    """
    Runs the evaluation for multiple models and context sizes.

    :param models: A set of model names to compare.
    :param context_sizes: A list of context window sizes to test.
    :return: A list of evaluation results.
    """
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

    return eval_results
