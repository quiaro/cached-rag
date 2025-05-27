from typing import Sequence, Tuple, Any, Optional
from langchain_core.documents import Document
from langchain_mongodb.docstores import MongoDBDocStore

# This is a custom byte store for MongoDB.
# The idea for this class came from LocalFileStore.
# LocalFileStore is used by CacheBackedEmbeddings so
# the idea was to create a drop-in replacement for LocalFileStore using MongoDB.
# MongoDBDocStore does not support the same API as LocalFileStore, so MongoDBByteStore 
# fills the gap between the two.
class MongoDBByteStore:
    def __init__(self, docstore: MongoDBDocStore):
        self.docstore = docstore

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        docs = self.docstore.mget(keys)
        result = []
        for doc in docs:
            if doc is None:
                result.append(None)
            else:
                # Assume page_content is str, encode to bytes
                result.append(doc.page_content.encode('utf-8'))
        return result

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]], batch_size: int = 100) -> None:
        # Store bytes as Document with page_content as str
        docs = [(k, Document(page_content=v.decode('utf-8'), metadata={})) for k, v in key_value_pairs]
        self.docstore.mset(docs, batch_size=batch_size)

    def __getattr__(self, name: str) -> Any:
        # Proxy all other attributes/methods to the underlying docstore
        return getattr(self.docstore, name)
