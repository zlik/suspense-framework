import os

import ollama
from dotenv import load_dotenv
from flask import Flask, redirect, render_template_string, request, session, url_for
from groq import Groq
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import CompletionMessage, UserMessage
from openai import OpenAI
from rag import add_document, retrieve_context

# Get the absolute path of the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Load API keys from .env file
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_STACK_API_KEY = os.getenv("LLAMA_STACK_API_KEY")
LLAMA_STACK_BASE_URL = os.getenv("LLAMA_STACK_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Ensure API keys are set
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env file.")
if not LLAMA_STACK_API_KEY or not LLAMA_STACK_BASE_URL:
    raise ValueError("Missing Llama Stack API configuration in .env file.")
if not OPENAI_API_KEY or not OPENAI_BASE_URL:
    raise ValueError("Missing OpenAI API configuration in .env file.")

app = Flask(__name__)
app.secret_key = "session_random_key"
app_model_ollama = "llama3"
app_model_groq = "llama3-8b-8192"
app_model_llama_stack = "llama3.3-70b-instruct"
app_model_openai = app_model_llama_stack

groq_client = Groq(api_key=GROQ_API_KEY)
llama_stack_client = LlamaStackClient(
    base_url=LLAMA_STACK_BASE_URL, api_key=LLAMA_STACK_API_KEY
)
openai_client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hagakure: A Sample GenAI Inference & RAG App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; }
        select { margin-bottom: 1rem; padding: 8px; }
        textarea { width: 100%; height: 150px; }
        button { padding: 8px 16px; font-size: 16px; }
        .response { margin-top: 20px; padding: 10px; background-color: #f0f0f0; }
        .conversation { margin-top: 20px; }
        .debug { margin-top: 20px; padding: 10px; background-color: #e0e0e0; font-family: monospace; }
    </style>
    <script>
        function changeProvider() {
            document.getElementById('providerForm').submit();
        }
    </script>
</head>
<body>
    <h2>Hagakure: A Sample GenAI Inference & RAG App</h2>
    <p>
        Powered by
        {% if session.get('provider') == 'groq' %}
            <a href="https://groq.com">Groq</a> ({{ app_model_groq }})
        {% elif session.get('provider') == 'llama_stack' %}
            <a href="https://llamastack.com">Llama Stack</a> ({{ app_model_llama_stack }})
        {% elif session.get('provider') == 'openai' %}
            <a href="https://openai.com">OpenAI</a> ({{ app_model_openai }})
        {% else %}
            <a href="https://github.com/ollama/ollama">Ollama</a> ({{ app_model_ollama }})
        {% endif %}
    </p>

    <form method="post" id="providerForm">
        <select name="provider" onchange="changeProvider()">
            <option value="groq" {% if session.get('provider', 'groq') == 'groq' %}selected{% endif %}>Groq</option>
            <option value="ollama" {% if session.get('provider') == 'ollama' %}selected{% endif %}>Ollama</option>
            <option value="llama_stack" {% if session.get('provider') == 'llama_stack' %}selected{% endif %}>Llama Stack</option>
            <option value="openai" {% if session.get('provider') == 'openai' %}selected{% endif %}>OpenAI</option>
        </select>
    </form>
    
    <form method="post">
        <textarea name="prompt" placeholder="Enter your prompt here"></textarea><br><br>
        <button type="submit">Submit</button>
        <button type="button" onclick="location.href='{{ url_for('reset') }}'">Reset Conversation</button>
    </form>

    <div class="conversation">
        {% for item in session.get(session.get('provider', 'groq') + '_conversation', []) %}
            <div><strong>You:</strong> {{ item['prompt'] }}</div>
            <div class="response"><strong>Model ({{ item['provider'] }}):</strong> {{ item['response'] }}</div>
            <hr>
        {% endfor %}
    </div>
    
    {% if session.get('debug_info') %}
    <div class="debug">
        <h3>Debug Information</h3>
        <pre>{{ session['debug_info'] }}</pre>
    </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    provider = session.get("provider", "groq")
    conversation_key = f"{provider}_conversation"
    context_key = f"{provider}_context"
    debug_info = ""
    response = ""

    if conversation_key not in session:
        session[conversation_key] = []

    if request.method == "POST":
        new_provider = request.form.get("provider")
        if new_provider and new_provider != provider:
            session["provider"] = new_provider
            session.pop("debug_info", None)  # Clear debug info when provider changes
            return redirect(url_for("index"))

        prompt = request.form.get("prompt")
        if prompt:
            # Retrieve relevant documents for RAG
            retrieved_context, rag_debug_info = retrieve_context(prompt)

            # Combine retrieved context with user input
            enhanced_prompt = f"Context:\n{retrieved_context}\n\nUser Prompt: {prompt}"
            debug_info += f"[INFO] Final prompt sent to LLM:\n{enhanced_prompt}\n"
            debug_info += f"[INFO] RAG Processing Details:\n{rag_debug_info}\n"

            if provider == "groq":
                context = session.get(context_key, [])
                context.append({"role": "user", "content": enhanced_prompt})

                debug_info += f"Calling Groq API: groq_client.chat.completions.create with parameters:\n"
                debug_info += f"  Model: {app_model_groq}\n"
                debug_info += f"  Messages: {context}\n"

                groq_response = groq_client.chat.completions.create(
                    messages=context,
                    model=app_model_groq,
                )
                response = groq_response.choices[0].message.content
                context.append({"role": "assistant", "content": response})
                session[context_key] = context

            elif provider == "ollama":
                context = session.get(context_key)
                debug_info += "Calling Ollama API\n"
                result = ollama.generate(
                    model=app_model_ollama, prompt=enhanced_prompt, context=context
                )
                response = result["response"]
                session[context_key] = result.get("context")

            elif provider == "llama_stack":
                context = session.get(context_key, [])
                user_message = {"role": "user", "content": enhanced_prompt}
                context.append(user_message)

                debug_info += f"Calling Llama API: llama_stack_client.inference.chat_completion with parameters:\n"
                debug_info += f"  Model: {app_model_llama_stack}\n"
                debug_info += f"  Prompt: {enhanced_prompt}\n"
                debug_info += f"  Context: {context}\n"

                llama_response = llama_stack_client.inference.chat_completion(
                    messages=context,
                    model_id=app_model_llama_stack,
                )
                response_text = llama_response.completion_message.content.text
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "stop_reason": llama_response.completion_message.stop_reason,
                }
                response = response_text
                context.append(assistant_message)
                session[context_key] = context

            elif provider == "openai":
                context = session.get(context_key, [])
                context.append({"role": "user", "content": enhanced_prompt})

                debug_info += f"Calling OpenAI API: openai_client.chat.completions.create with parameters:\n"
                debug_info += f"  Model: {app_model_openai}\n"
                debug_info += f"  Prompt: {enhanced_prompt}\n"
                debug_info += f"  Context: {context}\n"

                completion = openai_client.chat.completions.create(
                    model=app_model_openai,
                    messages=context,
                )
                response = completion.choices[0].message.content
                context.append({"role": "assistant", "content": response})
                session[context_key] = context

            session.setdefault(conversation_key, []).append(
                {"prompt": enhanced_prompt, "response": response, "provider": provider}
            )
            session["debug_info"] = debug_info

    else:
        session.pop("debug_info", None)  # Clear debug info when page is refreshed

    return render_template_string(
        HTML_TEMPLATE,
        app_model_ollama=app_model_ollama,
        app_model_groq=app_model_groq,
        app_model_llama_stack=app_model_llama_stack,
        app_model_openai=app_model_openai,
    )


@app.route("/reset")
def reset():
    provider = session.get("provider", "groq")
    session.pop(f"{provider}_conversation", None)
    session.pop(f"{provider}_context", None)
    session.pop("debug_info", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001)
