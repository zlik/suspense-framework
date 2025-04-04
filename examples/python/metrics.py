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
        start = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        end = time.time()

        response = completion.choices[0].message.content
        total_time = end - start
        token_count = completion.usage.total_tokens
        tps = token_count / total_time if total_time > 0 else 0

        results.append({
            "prompt": prompt,
            "response": response,
            "total_response_time": round(total_time, 3),
            "tokens": token_count,
            "tps": round(tps, 2),
        })

    return results

metrics = measure_metrics(prompts, model)

for m in metrics:
    print(m)
