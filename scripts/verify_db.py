"""
verify_db.py

This file connects to our local Chroma vector database to make sure
all the expected collections exist and contain the correct number of records.
It acts as a quick sanity check before we start querying the database.
"""

from __future__ import annotations

import sys
from pathlib import Path
import chromadb


def verify_collections() -> bool:
    """Connects to the ChromaDB database and verifies the count of each collection.

    It checks for the four expected collections (quran_dense, hadith_dense,
    tafsir_ar_dense, tafsir_en_dense) and prints their document counts.

    Returns:
        bool: True if all collections were checked successfully, False otherwise.
    """
    db_path = "./data/chroma_db"
    resolved_path = Path(db_path).resolve()
    print(f"Connecting to ChromaDB at: {resolved_path}")

    if not resolved_path.exists():
        print(f"Error: The database directory does not exist at {resolved_path}")
        return False

    try:
        # Initialize PersistentClient pointing to "./data/chroma_db"
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"Failed to initialize ChromaDB PersistentClient: {e}")
        return False

    expected_collections = [
        "quran_dense",
        "hadith_dense",
        "tafsir_ar_dense",
        "tafsir_en_dense"
    ]

    success = True
    for collection_name in expected_collections:
        try:
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            print(f"Collection '{collection_name}' count: {count:,} documents")
        except Exception as e:
            print(f"Error retrieving collection '{collection_name}': {e}")
            success = False

    return success


def main() -> int:
    """Runs the database verification pipeline and exits with status code.

    Returns:
        int: Exit status code (0 for success, 1 for failure).
    """
    success = verify_collections()
    if success:
        print("Database sanity check passed successfully.")
        return 0
    else:
        print("Database sanity check encountered issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
