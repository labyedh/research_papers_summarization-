import os
from dotenv import load_dotenv

load_dotenv()

# Elasticsearch Credentials
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY_ID = os.getenv("ELASTIC_API_KEY_ID")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

INDEX_NAME = "arxiv-articles"

# LLM Model
#LLM_MODEL_NAME = "tiiuae/falcon-11B"

EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
LLM_MODEL_NAME = "emredeveloper/DeepSeek-R1-Medical-COT"