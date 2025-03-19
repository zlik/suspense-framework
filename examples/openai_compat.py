import os

import openai

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#
#  USING LLAMA API
#  To us the Llama API comment out the USING OPENAI section above and uncomment this section
#
#  LLAMA Section BEGIN
# --------------------------------------------------------------------------------------------------
baseUrl = os.getenv("LLAMA_API_URL")
apiKey = os.getenv("LLAMA_API_KEY")
model = os.getenv("LLAMA_MODEL")
client = OpenAI(base_url=baseUrl, api_key=apiKey)
# --------------------------------------------------------------------------------------------------
#  LLAMA Section END

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        },
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue

    print(chunk.choices[0].delta.content, end="")
print()

# Response headers:
print("----- custom response headers test -----")
response = client.chat.completions.with_raw_response.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
)
completion = response.parse()
print(response.request_id)
print(completion.choices[0].message.content)
