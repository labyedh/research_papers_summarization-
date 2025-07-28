import json
from elastic_search_utils import connect_elasticsearch, check_index_exists, index_document, search_documents
from embedding_utils import generate_embedding
from llm_utils import generate_answer

def main():
    es = connect_elasticsearch()
    check_index_exists(es)

    # Load and index documents (only needed once)
    with open("arxiv_papers_with_pdf.json", "r") as f:
        articles = json.load(f)

    """for article in articles:
        doc = {
            "title": article["title"],
            "authors": article["authors"],
            "abstract": article["abstract"],
            "doi": article["doi"],
            "url": article["url"],
            "pdf_url": article["pdf_url"],
            "embedding": generate_embedding(article["abstract"])
        }
        index_document(es, doc)"""

    # User query
    query_text = "cancer detection using biosensors"
    query_embedding = generate_embedding(query_text)

    # Search in Elasticsearch
    retrieved_docs = search_documents(es, query_embedding)

    # Build context
    context = "\n\n".join(
        [f"Title: {doc['_source']['title']}\nAbstract: {doc['_source']['abstract']}\nPDF_LINK:{doc['_source']['pdf_url']}" for doc in retrieved_docs]
    )

    # Generate response
    llm_response = generate_answer(context,query_text)

    print("\nGenerated Answer:")
    print(llm_response)

if __name__ == "__main__":
    main()
