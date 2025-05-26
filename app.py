import os
import time
import logging
import chainlit as cl
import datetime
import hashlib
import argparse
from dotenv import load_dotenv
from operator import itemgetter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig

# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

"""
We will load our environment variables here.
"""
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = os.environ["QDRANT_PORT"]
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL))

os.environ["LANGCHAIN_PROJECT"] = f"Cached RAG - {timestamp}"
# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
text_loader = TextLoader("./data/paul_graham_short.txt")
documents = text_loader.load()

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

hf_embeddings = TogetherEmbeddings(
    model=HF_EMBED_ENDPOINT,
    api_key=TOGETHER_API_KEY,
)

safe_namespace = hashlib.md5(hf_embeddings.model.encode()).hexdigest()
collection_name = "paul_graham_essays"

store = LocalFileStore(f"./cache/{safe_namespace}/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    hf_embeddings, store, namespace=safe_namespace, batch_size=32
)


client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
# Check if collection exists before creating it
collections = client.get_collections().collections
if not any(c.name == collection_name for c in collections):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# Typical QDrant Vector Store Set-up
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=cached_embedder)

try:
    start_time = time.time()
    vectorstore.add_documents(split_documents)
    end_time = time.time()
    
    logger.debug(f"Time to add documents to Qdrant: {end_time - start_time:.2f} seconds")
    logger.info("Documents added to Qdrant successfully")
except Exception as e:
    logger.error(f"Error adding documents to Qdrant: {str(e)}")
    raise

mmr_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1})


# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

### 2. CREATE PROMPT TEMPLATE
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
hf_llm = ChatTogether(
    model=f"{HF_LLM_ENDPOINT}",
    max_tokens=512,
    top_p=0.95,
    temperature=0.01,
    request_timeout=20,
    api_key=TOGETHER_API_KEY,
)

@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "Paul Graham Essay Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = rag_prompt | hf_llm

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    # 1. Retrieve context using the retriever
    docs = mmr_retriever.get_relevant_documents(message.content)
    # If async: docs = await mmr_retriever.aget_relevant_documents(message.content)
    context = "\n\n".join([doc.page_content for doc in docs])

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content, "context": context},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # Try to extract the text from the chunk
        if hasattr(chunk, "content"):
            token = chunk.content
        else:
            token = str(chunk)
        await msg.stream_token(token)

    await msg.send()
