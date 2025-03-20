from tabulate import tabulate
from colorama import Fore, Style

def display_results(eval_results):
    """
    Displays evaluation results in a tabulated format with colored output.
    """
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
