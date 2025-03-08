import base64
import json
import logging
import time
from io import BytesIO

import ollama
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LargeContextEval:
    def __init__(self, model_name, max_context_tokens, test_cases):
        """
        Initialize evaluation with model name, context window size, and test cases.
        """
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.test_cases = test_cases
        self.results = []

    def run_eval(self):
        """
        Execute evaluation for each test case.
        """
        for i, test in enumerate(self.test_cases):
            prompt = test["input"]
            expected_output = test["expected_output"]
            input_image = test.get("image")
            logging.info(f"Running test {i+1}/{len(self.test_cases)}...")

            start_time = time.time()

            if input_image:
                response = self.generate_with_image(prompt, input_image)
            else:
                response = ollama.generate(model=self.model_name, prompt=prompt).get(
                    "response", ""
                )

            end_time = time.time()

            result = {
                "test_id": i + 1,
                "input_length": len(prompt.split()),
                "expected_output": expected_output,
                "actual_output": response,
                "response_time": end_time - start_time,
                "output_length": (
                    len(response.split()) if isinstance(response, str) else None
                ),
                "correct": expected_output in response,
            }
            self.results.append(result)
            logging.info(json.dumps(result, indent=2))

    def generate_with_image(self, prompt, image_path):
        """
        Generate response using both text and an image.
        """
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        response = ollama.generate(
            model=self.model_name, prompt=prompt, image=img_base64
        )
        return response

    def save_results(self, filename="eval_results.json"):
        """
        Save results to a JSON file.
        """
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Results saved to {filename}")

    def analyze_results(self):
        """
        Analyze and summarize evaluation results.
        """
        total_tests = len(self.results)
        success_count = sum(1 for r in self.results if r["correct"])
        avg_response_time = sum(r["response_time"] for r in self.results) / total_tests
        avg_output_length = (
            sum(
                r["output_length"]
                for r in self.results
                if r["output_length"] is not None
            )
            / total_tests
        )

        summary = {
            "total_tests": total_tests,
            "success_count": success_count,
            "success_rate": success_count / total_tests,
            "avg_response_time": avg_response_time,
            "avg_output_length": avg_output_length,
        }
        logging.info("Evaluation Summary:")
        logging.info(json.dumps(summary, indent=2))
        return summary


# Example test cases focusing on long-range dependencies with text and images
test_cases = [
    {
        "input": "This is a long passage that contains a critical detail near the start. Remember: 'The secret code is 7461'. Now, let's add several paragraphs of unrelated text... (Imagine 3K+ words here) ... Finally, what is the secret code?",
        "expected_output": "7461",
    },
    {
        "input": "A scientific paper discussing quantum mechanics starts with a premise. (Imagine a long detailed explanation) ... At the end, summarize the first key argument mentioned.",
        "expected_output": "(First key argument)",
    },
    {
        "input": "Describe what you see in this image and infer the possible context.",
        "image": "example_image.jpg",
        "expected_output": "(Relevant description)",
    },
]

if __name__ == "__main__":
    eval_runner = LargeContextEval(
        model_name="llama3", max_context_tokens=8000, test_cases=test_cases
    )
    eval_runner.run_eval()
    eval_runner.save_results()
    eval_runner.analyze_results()
