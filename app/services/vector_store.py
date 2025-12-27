import os
import threading
from typing import List, Dict, Optional, Any
import logging

from chromadb import PersistentClient, EphemeralClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "legal_docs",
        persist_directory: Optional[str] = "./chroma_db",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        allow_reset: bool = False,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self._lock = threading.Lock()
        self._embed_model: Optional[SentenceTransformer] = None

        try:
            if persist_directory:
                os.makedirs(persist_directory, exist_ok=True)
                self.client = PersistentClient(path=persist_directory)
                logger.info(f"Chroma PersistentClient initialized at {persist_directory}")
            else:
                self.client = EphemeralClient()
                logger.info("Chroma EphemeralClient initialized (in-memory)")
        except Exception:
            logger.exception("Failed to initialize Chroma client")
            raise

        try:
            existing = [c.name for c in self.client.list_collections()]
            if collection_name in existing:
                if allow_reset:
                    logger.warning(f"Resetting existing collection '{collection_name}'")
                    self.client.delete_collection(name=collection_name)
                    self.collection = self.client.create_collection(name=collection_name)
                else:
                    self.collection = self.client.get_collection(name=collection_name)
            else:
                self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Using Chroma collection: '{collection_name}'")
        except Exception:
            logger.exception("Failed to get/create Chroma collection")
            raise

    def _load_embedding_model(self):
        if self._embed_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            try:
                self._embed_model = SentenceTransformer(self.embedding_model_name)
            except Exception:
                logger.exception("Failed to load embedding model")
                raise
        return self._embed_model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        model = self._load_embedding_model()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.extend(embs.tolist())
        return embeddings

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 64,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in documents]
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, metadatas must have same length")

        with self._lock:
            for i in range(0, len(documents), batch_size):
                b_ids = ids[i : i + batch_size]
                b_docs = documents[i : i + batch_size]
                b_meta = metadatas[i : i + batch_size]

                embeddings = self.embed_texts(b_docs)
                self.collection.add(
                    ids=b_ids,
                    documents=b_docs,
                    metadatas=b_meta,
                    embeddings=embeddings,
                )
                logger.info(f"Added batch of {len(b_docs)} docs to {self.collection_name}")

    def upsert_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 64,
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in documents]
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("ids, documents, metadatas must match")

        with self._lock:
            for i in range(0, len(documents), batch_size):
                b_ids = ids[i : i + batch_size]
                b_docs = documents[i : i + batch_size]
                b_meta = metadatas[i : i + batch_size]

                embeddings = self.embed_texts(b_docs)
                self.collection.upsert(
                    ids=b_ids,
                    documents=b_docs,
                    metadatas=b_meta,
                    embeddings=embeddings,
                )
                logger.info(f"Upserted batch of {len(b_docs)} docs")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Allowed include keys in chroma v0.5+
        allowed_includes = {"documents", "embeddings", "metadatas", "distances", "uris", "data"}
        if include is None:
            include = ["documents", "metadatas", "distances"]
        include = [i for i in include if i in allowed_includes]

        query_embedding = self.embed_texts([query_text])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=include,
        )

        def _first_or_empty(field_name):
            val = results.get(field_name, [])
            if not val:
                return []
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
                return val[0]
            return val

        documents = _first_or_empty("documents")
        metadatas = _first_or_empty("metadatas")
        distances = _first_or_empty("distances")
        ids = _first_or_empty("ids")

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }

    def delete_by_id(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
        logger.info(f"Deleted IDs: {ids}")

    def reset_collection(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        logger.warning(f"Collection '{self.collection_name}' reset!")

    def get_collection_stats(self) -> Dict[str, Any]:
        return {"count": self.collection.count()}

    def add_single(self, id_: str, document: str, metadata: Optional[Dict[str, Any]] = None):
        self.add_documents([id_], [document], [metadata or {}])

