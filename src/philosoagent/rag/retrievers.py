from loguru import logger
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever

from .embedding import get_embedding_model
from src.philosoagent.config import settings

Retriever = MongoDBAtlasHybridSearchRetriever

def get_retriever(
        embedding_model_id: str,
        k: int = 3,
        device: str = "cpu"
    ) -> Retriever | None:
    """Creates and returns a hybrid search retriever with the specified embedding model.

    Args:
        embedding_model_id (str): The identifier for the embedding model to use.
        k (int, optional): Number of documents to retrieve. Defaults to 3.
        device (str, optional): Device to run the embedding model on. Defaults to "cpu".

    Returns:
        Retriever | None: A configured hybrid search retriever or None if initialization fails.
    """

    logger.info(
        f"Initializing retriever | model: {embedding_model_id} | device: {device} | top_k: {k}"
    )

    embedding_model = get_embedding_model(embedding_model_id, device)

    return get_hybrid_search_retriever(k, embedding_model)

def get_hybrid_search_retriever(
        k: int,
        embedding_model: HuggingFaceEmbeddings | None = None
    ) -> MongoDBAtlasHybridSearchRetriever | None:
    """Creates a MongoDB Atlas hybrid search retriever with the given embedding model.

    Args:
        embedding_model (HuggingFaceEmbeddings): The embedding model to use for vector search.
        k (int): Number of documents to retrieve.

    Returns:
        MongoDBAtlasHybridSearchRetriever | None: A configured hybrid search retriever using both
            vector and text search capabilities, or None if initialization fails.
    """

    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=settings.MONGO_URI,
            embedding=embedding_model,
            namespace=f"{settings.MONGO_DB_NAME}.{settings.MONGO_LONG_TERM_MEMORY_COLLECTION}",
            text_key="chunk",
            embedding_key="embedding",
            relevance_score_fn="dotProduct",
        )
    
    retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vectorstore,
            search_index_name="hybrid_search_index",
            top_k=k,
            vector_penalty=50,
            fulltext_penalty=50,
        )

    return retriever