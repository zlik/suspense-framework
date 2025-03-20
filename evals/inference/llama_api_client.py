from llama_stack_client import LlamaStackClient
from config import getenv

# Load API credentials
LLAMA_API_KEY = getenv("LLAMA_API_KEY")
LLAMA_API_BASE_URL = getenv("LLAMA_API_BASE_URL")

# Initialize Llama client
llama_client = LlamaStackClient(base_url=LLAMA_API_BASE_URL, api_key=LLAMA_API_KEY)

def query_llama_api(prompt, model="llama3.3-70b-llama_api", max_tokens=2048):
    """Function to query the Llama API with a specified model and context length."""
    truncated_prompt = prompt[:max_tokens]  # Truncate prompt to fit context window
    try:
        response = llama_client.inference.chat_completion(
            model_id=model,
            messages=[{"role": "user", "content": truncated_prompt}],
        )
        return response.completion_message.content.text if response else "Error: API request failed"
    except Exception as e:
        return f"Error: {str(e)}"
