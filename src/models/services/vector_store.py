import os
import uuid
from datetime import datetime
from typing import TypedDict, Optional

import chromadb
from dataclasses import dataclass
from chromadb import GetResult
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from numpy import ndarray
from pydantic import BaseModel
from enum import Enum



class SupportedFileType(str, Enum):
    txt = "text/plain"
    markdown = "text/markdown"
    rtf="text/rtf"
    pdf="application/pdf"

class DocumentMetadata(TypedDict):
    filename: str
    filetype: str
    created_at: str

@dataclass
class StoreDocument:
    id: str
    text: str
    embedding: Optional[ndarray]
    metadata: DocumentMetadata

class StoreFullFile(BaseModel):
    filename: str
    filetype: str
    created_at: datetime

class VectorStore:
    # chroma_client = chromadb.CloudClient(
    #     api_key=os.getenv("CHROMADB_API_KEY"),
    #     database=os.getenv("VECTOR_STORE_DATABASE")
    # )
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST")
    )

    embedding_function = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
        dimensions=1024
    )

    @staticmethod
    def init():
        VectorStore.chroma_client.get_or_create_collection(os.getenv("VECTOR_STORE_COLLECTION"), embedding_function=VectorStore.embedding_function)

    @staticmethod
    def semantic_search(collection_name: str, texts: list[str], accept_threshold: float = 0.70, n_results: int = 3) -> list[list[StoreDocument]]:
        """ Makes a semantic search in the vector store
            :arg collection_name The name of the collection in the vector store
            :arg texts List of queries
            :arg accept_threshold Threshold above which a result will be included in the response
            :arg n_results The number of results in the response
         """
        collection = VectorStore.chroma_client.get_collection(collection_name, embedding_function=VectorStore.embedding_function)
        query_results = collection.query(
            query_texts=texts,
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )

        final_results: list[list[StoreDocument]] = []
        for index in range(len(query_results["documents"])):
            results: list[StoreDocument] = []
            for id, distance, document, metadata in zip(query_results["ids"][index], query_results["distances"][index], query_results["documents"][index], query_results["metadatas"][index]):
                if distance < accept_threshold:
                    continue
                results.append(StoreDocument(id=id, text=document, embedding=None, metadata=DocumentMetadata(**metadata)))
            final_results.append(results)

        return final_results

    @staticmethod
    def get_all_docs(collection_name: str, limit: int = 100, include_embeddings = False) -> list[StoreDocument]:
        collection = VectorStore.chroma_client.get_collection(collection_name)
        includes = ["documents", "metadatas"]
        if include_embeddings:
            includes += ["embeddings"]

        docs: GetResult = collection.get(limit=limit, include=includes)

        return [
            StoreDocument(
                id=docs["ids"][index],
                text=docs["documents"][index],
                embedding=docs["embeddings"][index] if docs.get("embeddings") is not None else None,
                metadata=DocumentMetadata(**docs["metadatas"][index])
            ) for index, doc in enumerate(docs["ids"])
        ]

    @staticmethod
    def get_docs_by_ids(ids: list[str], collection_name: str, include_embeddings = False) -> list[StoreDocument]:
        collection = VectorStore.chroma_client.get_collection(collection_name)
        includes = ["documents", "metadatas"]
        if include_embeddings:
            includes += ["embeddings"]

        docs: GetResult = collection.get(ids=ids, include=includes)
        return [
            StoreDocument(
                id=docs["ids"][index],
                text=docs["documents"][index],
                embedding=docs["embeddings"][index] if docs.get("embeddings") is not None else None,
                metadata=DocumentMetadata(**docs["metadatas"][index])
            ) for index, doc in enumerate(docs)
        ]

    @staticmethod
    def get_docs_by_filename(filename: str, collection_name: str, include_embeddings = False) -> list[StoreDocument]:
        collection = VectorStore.chroma_client.get_collection(collection_name)
        includes = ["documents", "metadatas"]
        if include_embeddings:
            includes += ["embeddings"]

        docs: GetResult = collection.get(where={"filename": filename}, include=includes)

        return [
            StoreDocument(
                id=docs["ids"][index],
                text=docs["documents"][index],
                embedding=docs["embeddings"][index] if docs.get("embeddings") is not None else None,
                metadata=DocumentMetadata(**docs["metadatas"][index])
            ) for index, doc in enumerate(docs["documents"])
        ]

    @staticmethod
    def insert_docs(docs: list[str], collection_name: str, filename: str, filetype: str, custom_ids: list[str] | None = None):
        if custom_ids and len(docs) != len(custom_ids):
            raise ValueError("Custom ids should be the same amount as the docs")

        collection = VectorStore.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=VectorStore.embedding_function
        )

        docs_to_store: list[StoreDocument] = []

        for index, doc in enumerate(docs):
            docs_to_store.append(
                StoreDocument(
                    id=custom_ids[index] if custom_ids else str(uuid.uuid4()),
                    text=doc,
                    embedding=None,
                    metadata=DocumentMetadata(filename=filename, filetype=filetype, created_at=str(datetime.now()))
                )
            )

        collection.add(
            ids=[doc.id for doc in docs_to_store],
            documents=[doc.text for doc in docs_to_store],
            metadatas=[dict(doc.metadata) for doc in docs_to_store]
        )

    @staticmethod
    def update_docs(updated_docs: list[StoreDocument], collection_name: str):
        collection = VectorStore.chroma_client.get_collection(collection_name)
        collection.update(
            ids=[doc.id for doc in updated_docs],
            documents=[doc.text for doc in updated_docs],
            metadatas=[dict(doc.metadata) for doc in updated_docs]
        )

    @staticmethod
    def delete_docs_by_ids(ids: list[str], collection_name: str):
        collection = VectorStore.chroma_client.get_collection(collection_name)
        collection.delete(
            ids=ids
        )

    @staticmethod
    def delete_document_by_name(filename: str, collection_name: str):
        collection = VectorStore.chroma_client.get_collection(collection_name)
        collection.delete(
            where={"filename": filename}
        )


