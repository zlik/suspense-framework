import ollama
from flask import Flask, redirect, render_template_string, request, session, url_for
from groq import Groq

app = Flask(__name__)
app.secret_key = "session_random_key"
app_model_ollama = "llama3"
app_model_groq = "llama3-8b-8192"
groq_client = Groq(api_key="<USE_YOUR_GROQ_KEY_HERE>")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GenAI Inference Sample App</title>
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
    <h2>GenAI Inference Sample App</h2>
    <p>
        Powered by
        {% if session.get('provider') == 'groq' %}
            <a href="https://groq.com">Groq</a> ({{ app_model_groq }})
        {% else %}
            <a href="https://github.com/ollama/ollama">Ollama</a> ({{ app_model_ollama }})
        {% endif %}
    </p>

    <form method="post" id="providerForm">
        <select name="provider" onchange="changeProvider()">
            <option value="groq" {% if session.get('provider', 'groq') == 'groq' %}selected{% endif %}>Groq</option>
            <option value="ollama" {% if session.get('provider') == 'ollama' %}selected{% endif %}>Ollama</option>
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
            if provider == "groq":
                context = session.get(context_key, [])
                context.append({"role": "user", "content": prompt})

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
            else:
                context = session.get(context_key)
                debug_info += f"Calling Ollama API: ollama.generate with parameters:\n"
                debug_info += f"  Model: {app_model_ollama}\n"
                debug_info += f"  Prompt: {prompt}\n"
                debug_info += f"  Context: {context}\n"

                result = ollama.generate(
                    model=app_model_ollama, prompt=prompt, context=context
                )
                response = result["response"]
                session[context_key] = result.get("context")

            session.setdefault(conversation_key, []).append(
                {"prompt": prompt, "response": response, "provider": provider}
            )
            session["debug_info"] = debug_info

    else:
        session.pop("debug_info", None)  # Clear debug info when page is refreshed

    return render_template_string(
        HTML_TEMPLATE, app_model_ollama=app_model_ollama, app_model_groq=app_model_groq
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
