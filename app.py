from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from elastic_search_utils import connect_elasticsearch, check_index_exists, search_documents
from embedding_utils import generate_embedding
from llm_utils import generate_answer

app = Flask(__name__)
CORS(app)  # Enable CORS if needed for frontend integration

# Initialize Elasticsearch connection when app starts
es = connect_elasticsearch()
if es is not None:
    check_index_exists(es)
else:
    raise RuntimeError("Failed to connect to Elasticsearch")

@app.route('/query', methods=['POST'])
def handle_query():
    # Get query from request
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400
    
    query_text = data['query']
    
    try:
        # Generate embedding
        query_embedding = generate_embedding(query_text)
        
        # Search documents
        retrieved_docs = search_documents(es, query_embedding)
        
        # Build context
        context = "\n\n".join(
            [f"Title: {doc['_source']['title']}\nAbstract: {doc['_source']['abstract']}\nPDF_LINK:{doc['_source']['pdf_url']}" 
             for doc in retrieved_docs]
        )
        
        # Generate answer
        llm_response = generate_answer(context, query_text)
        
        return jsonify({
            "query": query_text,
            "answer": llm_response,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)