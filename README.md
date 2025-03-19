# Suspense Framework
Suspense is a GenAI experimentation framework.

## Features
- **Hagakure**: A Sample GenAI Inference & RAG App
- **Evals**: Coming soon

## Hagakure

### Overview

Hagakure is a simple GenAI inference web application that allows users to interact with multiple AI models via different providers, including:

* Groq
* Llama Stack
* OpenAI
* Ollama

The application also integrates Retrieval-Augmented Generation (RAG) using [Faiss](https://github.com/facebookresearch/faiss) for document retrieval.
Hagakure is built using [flask](https://github.com/pallets/flask).

### Features

* Multi-provider AI inference: Switch between AI providers dynamically
* Retrieval-Augmented Generation (RAG): Uses Faiss to retrieve relevant documents
* Session-based conversation storage: Keeps track of interactions per provider
* Simple web interface: Submit prompts and view responses easily

### Running the Application

#### 1. Populate the Knowledge Base

Run the following script to add sample documents to the Faiss index:

`python hagakure/rag_ingest.py`

#### 2. Start the Flask App

Run the web application:

`python hagakure/app.py`

By default, the application runs on http://127.0.0.1:5001/.

## Contributing

Feel free to fork this project and submit pull requests for improvements.

## License

This project is open-source under the MIT License.