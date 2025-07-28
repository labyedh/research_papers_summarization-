from elasticsearch import Elasticsearch
from config import ELASTIC_CLOUD_ID, ELASTIC_API_KEY_ID, ELASTIC_API_KEY, INDEX_NAME

def connect_elasticsearch():
    try:
        api_key = (ELASTIC_API_KEY_ID, ELASTIC_API_KEY)
        es = Elasticsearch(cloud_id=ELASTIC_CLOUD_ID, api_key=api_key)
        return es
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")
        return None

def check_index_exists(es):
    """Check if the index exists, if not, create it."""
    if not es.indices.exists(index=INDEX_NAME):
        index_mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "authors": {"type": "keyword"},
                "abstract": {"type": "text"},
                "doi": {"type": "keyword"},
                "url": {"type": "keyword"},
                "pdf_url": {"type": "keyword"},
                "embedding": {"type": "dense_vector", "dims": 768}
                            }
            }
        }

        es.indices.create(index=INDEX_NAME, body=index_mapping)
        print(f"Created index: {INDEX_NAME}")
    else:
        print(f"Index {INDEX_NAME} already exists.")

def index_document(es, doc):
    """Index a document in Elasticsearch."""
    es.index(index=INDEX_NAME, body=doc)

def search_documents(es, query_embedding, size=5):
    search_query = {
        "knn": [
            {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": size,
                "num_candidates": 50
            }
        ],
        "size": size
    }

    response = es.search(index=INDEX_NAME, body=search_query)
    return response["hits"]["hits"]

