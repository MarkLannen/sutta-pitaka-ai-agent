"""Vector store management using ChromaDB with Ollama embeddings."""

from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..config import CHROMA_PATH, OLLAMA_BASE_URL, EMBED_MODEL


class VectorStoreManager:
    """Manage ChromaDB vector store for sutta embeddings."""

    def __init__(
        self,
        collection_name: str = "pali_canon",
        persist_dir: Optional[Path] = None,
        embed_model: Optional[str] = None,
    ):
        """
        Initialize the vector store manager.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory for persistent storage (default: from config)
            embed_model: Ollama embedding model name (default: from config)
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir or CHROMA_PATH
        self.embed_model_name = embed_model or EMBED_MODEL

        # Ensure persist directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embed_model = OllamaEmbedding(
            model_name=self.embed_model_name,
            base_url=OLLAMA_BASE_URL,
        )

        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Pali Canon sutta embeddings"}
        )

        # Initialize vector store
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )

        # Storage context for LlamaIndex
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # Index (lazily initialized)
        self._index: Optional[VectorStoreIndex] = None

    @property
    def index(self) -> VectorStoreIndex:
        """Get or create the vector store index."""
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model,
            )
        return self._index

    def add_documents(
        self,
        documents: list[Document],
        show_progress: bool = True,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of LlamaIndex Documents to add
            show_progress: Whether to show progress bar
        """
        if not documents:
            return

        # Create index from documents (this embeds and stores them)
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=show_progress,
        )

    def get_document_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate the collection
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Pali Canon sutta embeddings"}
        )
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.collection
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self._index = None

    def collection_exists(self) -> bool:
        """Check if the collection has any documents."""
        return self.get_document_count() > 0
