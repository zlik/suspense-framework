from rag import add_document

# Add knowledge base documents
add_document(
    "Python is a programming language widely used for AI and machine learning."
)
add_document(
    "FAISS is a library developed by Facebook AI Research for fast similarity search."
)
add_document("OpenAI developed the GPT models which power modern chat applications.")

print("Documents added successfully.")
