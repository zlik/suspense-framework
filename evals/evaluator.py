def simple_evaluator(response, expected):
    """A simple evaluator that scores based on exact match and length similarity."""
    if response.strip().lower() == expected.strip().lower():
        return 1.0
    return max(0, 1 - abs(len(response) - len(expected)) / max(len(response), len(expected), 1))
