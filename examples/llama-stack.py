from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import UserMessage
from llama_stack_client.lib.inference.event_logger import EventLogger

import os
import pprint

load_dotenv()  # Load environment variables from .env

client = LlamaStackClient(
    base_url=os.getenv('API_BASE_URL'),
    api_key=os.getenv('API_KEY'),
  )

response = client.models.list()
pprint.pprint(response)

response = client.inference.chat_completion(
    messages=[
        UserMessage(
            role="user",
            content="Give me a JSON list of the top 5 countries to visit in 2025",
        )
    ],
    model_id="llama3.3-70b-instruct",
    stream=False,
)

pprint.pprint(response)

response = client.inference.chat_completion(
        messages=[
            UserMessage(
                role="user",
                content="Give me a JSON list of the top 5 countries to visit in 2025",
            )
        ],
    model_id="llama3.3-70b-instruct",
    stream=True,
)

for chunk in response:
    print(chunk.event.delta.text, end='')
