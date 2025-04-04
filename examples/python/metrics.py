import time
from openai import OpenAI

base_url = "https://api.openai.com/v1/"
api_key = API_KEY_HERE
model = "gpt-4o"

client = OpenAI(api_key=api_key, base_url=base_url)

prompts = [
    "Which planet do humans live on?",
    "Tell me a fun fact about octopuses.",
    "What's the capital of Japan?",
]

def measure_metrics(prompts, model):
    results = []

    for prompt in prompts:
        start_time = time.time()
        first_token_time = None
        full_response = ""

        # Use streaming to capture TTFT
        for chunk in client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            if not first_token_time:
                first_token_time = time.time()
            delta = chunk.choices[0].delta
            content = delta.content if delta and delta.content else ""
            full_response += content

        end_time = time.time()

        ttft = round(first_token_time - start_time, 3) if first_token_time else None
        total_time = round(end_time - start_time, 3)
        token_count = len(full_response.split())  # Rough estimate, or use tokenizer if needed
        tps = round(token_count / total_time, 2) if total_time > 0 else 0

        results.append({
            "prompt": prompt,
            "response": full_response,
            "ttft": ttft,
            "total_response_time": total_time,
            "tokens": token_count,
            "tps": tps,
        })

    return results

metrics = measure_metrics(prompts, model)

for m in metrics:
    print(m)
