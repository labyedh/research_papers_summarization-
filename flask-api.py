from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import torch
from elastic_search_utils import connect_elasticsearch, search_documents
from embedding_utils import generate_embedding
from llm_utils import generate_answer
from config import INDEX_NAME

app = Flask(__name__)

# Initialize Elasticsearch connection
es = connect_elasticsearch()

# Error handler for general exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        "error": str(e),
        "type": type(e).__name__
    }), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    if es is None:
        return jsonify({"status": "error", "message": "Elasticsearch connection failed"}), 500
    return jsonify({"status": "healthy", "elasticsearch": "connected"}), 200

# Search endpoint
@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query_text = data["query"]
        size = data.get("size", 5)  # Optional parameter for number of results

        # Generate embedding for the query
        query_embedding = generate_embedding(query_text)

        # Search documents
        results = search_documents(es, query_embedding, size=size)

        # Format response
        formatted_results = [{
            "title": doc["_source"]["title"],
            "authors": doc["_source"]["authors"],
            "abstract": doc["_source"]["abstract"],
            "doi": doc["_source"]["doi"],
            "url": doc["_source"]["url"],
            "pdf_url": doc["_source"]["pdf_url"],
        } for doc in results]

        return jsonify({
            "query": query_text,
            "results": formatted_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Question answering endpoint
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query_text = data["query"]
        size = data.get("size", 5)  # Number of documents to use as context

        # Generate embedding and search
        query_embedding = generate_embedding(query_text)
        retrieved_docs = search_documents(es, query_embedding, size=size)

        # Build context from retrieved documents
        context = "\n\n".join([
            f"Title: {doc['_source']['title']}\n"
            f"Abstract: {doc['_source']['abstract']}\n"
            f"PDF_LINK: {doc['_source']['pdf_url']}"
            for doc in retrieved_docs
        ])

        # Generate answer
        answer = generate_answer(context, query_text)

        return jsonify({
            "query": query_text,
            "answer": answer,
            "sources": [{
                "title": doc["_source"]["title"],
                "url": doc["_source"]["url"],
                "pdf_url": doc["_source"]["pdf_url"]
            } for doc in retrieved_docs]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Document indexing endpoint
@app.route("/index", methods=["POST"])
def index_document():
    try:
        doc = request.get_json()
        if not doc or "abstract" not in doc or "title" not in doc:
            return jsonify({"error": "Missing required fields in document"}), 400

        # Generate embedding for the document
        doc["embedding"] = generate_embedding(doc["abstract"])

        # Index the document
        result = es.index(index=INDEX_NAME, body=doc)
        return jsonify({
            "message": "Document indexed successfully",
            "id": result["_id"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
