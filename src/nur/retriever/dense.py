"""
dense.py

This file handles the semantic (dense) search. It connects to the local ChromaDB
database to find Quranic verses or Hadiths that match the meaning of the user's question,
even if the exact words are different.
"""

from __future__ import annotations

from typing import Any

import chromadb


class DenseRetriever:
    """DenseRetriever manages the connection to ChromaDB and performs

    semantic vector searches across the 4 Islamic text collections.
    """

    def __init__(self, db_path: str = "./data/chroma_db") -> None:
        """Connects to the local ChromaDB database and loads the collections.

        Args:
            db_path: The file path to the ChromaDB directory.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        self.collections = {
            "quran": self.client.get_collection("quran_dense"),
            "hadith": self.client.get_collection("hadith_dense"),
            "tafsir_ar": self.client.get_collection("tafsir_ar_dense"),
            "tafsir_en": self.client.get_collection("tafsir_en_dense"),
        }
        print("✅ DenseRetriever connected to ChromaDB.")

    def search(
        self, query_vector: list[float], source: str, top_k: int = 30
    ) -> list[dict[str, Any]]:
        """Searches a specific collection for the most similar chunks to the query vector.

        Args:
            query_vector: The dense vector representation of the user's query.
            source: The collection to search ('quran', 'hadith', 'tafsir_ar', 'tafsir_en').
            top_k: The maximum number of results to return.

        Returns:
            A list of matching chunks with their metadata and similarity score.
        """
        if source not in self.collections:
            raise ValueError(
                f"Source '{source}' not found. Available: {list(self.collections.keys())}"
            )

        collection = self.collections[source]

        results = collection.query(
            query_embeddings=[query_vector], n_results=top_k, include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                # Convert ChromaDB distance to a 0-1 similarity score
                similarity = 1.0 - (distance / 2.0)

                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": similarity,
                    }
                )

        return formatted_results
