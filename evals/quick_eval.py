from evaluator_runner import run_eval  # Import the evaluation function
from output.display import display_results  # Import the display function

# Define different context window lengths to test
context_sizes = [128, 256]  # Varying context lengths

if __name__ == "__main__":
    models_to_compare = {"llama3.3-8b-llama_api", "llama3.3-70b-llama_api"}

    # Run the evaluation and display results
    eval_results = run_eval(models=models_to_compare, context_sizes=context_sizes)
    display_results(eval_results)
