import os
import shutil
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

QDRANT_HOST = os.environ.get("QDRANT_HOST")
QDRANT_PORT = os.environ.get("QDRANT_PORT")
COLLECTION_NAME = "paul_graham_essays"
CACHE_DIR = "./cache"

# Remove Qdrant collection
def remove_qdrant_collection():
    if not QDRANT_HOST or not QDRANT_PORT:
        print("QDRANT_HOST or QDRANT_PORT not set in environment.")
        return
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' removed from Qdrant.")
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist in Qdrant.")
    except Exception as e:
        print(f"Error removing collection from Qdrant: {e}")

# Clear cache directory
def clear_cache_dir():
    if not os.path.exists(CACHE_DIR):
        print(f"Cache directory '{CACHE_DIR}' does not exist.")
        return
    try:
        for entry in os.listdir(CACHE_DIR):
            entry_path = os.path.join(CACHE_DIR, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        print(f"Cache directory '{CACHE_DIR}' cleared.")
    except Exception as e:
        print(f"Error clearing cache directory: {e}")

if __name__ == "__main__":
    remove_qdrant_collection()
    clear_cache_dir() 