import numpy as np
import json
import os
from numpy import float32, int32, int64, ndarray
from typing import Any, Optional, Dict, List, Union

#import sqlite3
#import faiss


class VectorDatabase:
    def __init__(self, data_dir: str="vector_db", collection: str="default", dim: int=128) -> None:
        self.data_dir = data_dir
        self.collection = collection
        self.dim = dim  # Vector dimension
        self.collection_path = os.path.join(data_dir, collection)
        os.makedirs(self.collection_path, exist_ok=True)

        self.vectors_file = os.path.join(self.collection_path, "vectors.npy")
        self.metadata_file = os.path.join(self.collection_path, "metadata.json")

        self.vectors = self.load_vectors()
        self.metadata = self.load_metadata()

    def load_vectors(self) -> ndarray:
        if os.path.exists(self.vectors_file):
            vectors = np.load(self.vectors_file)
            if vectors.shape[1] != self.dim:
                raise ValueError(f"Dimension mismatch: expected {self.dim}, found {vectors.shape[1]}")
            return vectors
        return np.empty((0, self.dim), dtype=np.float32)

    def load_metadata(self) -> List[Dict[str, str]]:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "rb") as f:
                    return json.loads(f.read())
            except json.JSONDecodeError:
                print(f"Warning: Corrupted metadata file for collection {self.collection}, resetting.")
                return []
        return []

    def save_vectors(self) -> None:
        if self.vectors.size > 0:
            np.save(self.vectors_file, self.vectors)

    def save_metadata(self) -> None:
        with open(self.metadata_file, "wb") as f:
            f.write(json.dumps(self.metadata))

    def add_vector(self, vector: ndarray, metadata: Dict[str, str]) -> None:
        """Add a single vector while ensuring correct dimension."""
        vector = np.array(vector, dtype=np.float32)
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vector.shape[0]}")

        if self.vectors.size == 0:
            self.vectors = np.array([vector], dtype=np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        self.metadata.append(metadata)

        self.save_vectors()
        self.save_metadata()

    def add_vector_binary(self, vector: ndarray, metadata: Dict[str, str]) -> None:
        vector = np.array(vector, dtype=np.uint8)
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {vector.shape[0]}")

        if self.vectors.size == 0:
            self.vectors = np.array([vector], dtype=np.uint8)
        else:
            self.vectors = np.vstack([self.vectors, vector])
        self.metadata.append(metadata)

        self.save_vectors()
        self.save_metadata()


    def batch_add_vectors(self, vectors, metadatas):
        """Efficient batch addition of multiple vectors."""
        vectors = np.array(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector batch dimension mismatch: expected {self.dim}, got {vectors.shape[1]}")

        if self.vectors.size == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.metadata.extend(metadatas)

        self.save_vectors()
        self.save_metadata()


    def binary_hamming_distance_search(self,query_vector: ndarray, k: int=2, threshold: int=160, filter_key: None=None, filter_value: None=None) -> List[Dict[str, Union[int32, int64, Dict[str, str]]]]:
        if self.vectors.size == 0:
            return []

        query_vector = np.array(query_vector, dtype=np.uint8)
        if query_vector.ndim != 1:
            raise ValueError("Query vector must be 1-dimensional.")

        if self.vectors.ndim != 2:
            raise ValueError("Vectors must be a 2-dimensional array.")

        if query_vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query vector and vectors must have the same dimension.")

        # Compute Hamming distances
        distances = np.count_nonzero(query_vector != self.vectors, axis=1)

        valid_indices = np.arange(len(distances))
        sorted_indices = np.argsort(distances)
        results = []
        for i in sorted_indices:
            if distances[i] > threshold or len(results) >= k:
                break
            metadata_item = self.metadata[i]
            vector_id = valid_indices[i]

            if filter_key is None or (filter_key in metadata_item and metadata_item[filter_key] == filter_value):
                results.append({
                    "id": vector_id,
                    "distance": distances[i],
                    "metadata": metadata_item,
                })
        if len(results):
           return results
        else:
          return None
          
          
    def search_vectors(self, query_vector: ndarray, k: int=5, threshold: float=0.5, filter_key: None=None, filter_value: None=None) -> List[Dict[str, Union[int64, float32, Dict[str, str]]]]:
        """Find the top-K closest vectors using NumPy-based cosine similarity."""
        if self.vectors.size == 0:
            return []

        query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dim}, got {query_vector.shape[0]}")

        # Compute cosine similarity
        dot_product = np.dot(self.vectors, query_vector)
        norm_query = np.linalg.norm(query_vector)
        norms = np.linalg.norm(self.vectors, axis=1)
        similarities = dot_product / (norm_query * norms + 1e-8)  # Avoid division by zero

        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        results = []

        for i in sorted_indices:
#            print(similarities[i])
            if similarities[i] < threshold or len(results) >= k:
                break
            metadata = self.metadata[i]
            if filter_key is None or (filter_key in metadata and metadata[filter_key] == filter_value):
                results.append({
                    "id": i,
                    "similarity": similarities[i],
                    "metadata": metadata,
                })
        if len(results):
           return results
        else:
          return None

    def get(self, where: Optional[Dict[str, int]]=None, incl: Optional[List[str]]=None) -> List[Any]:
        """Retrieve stored vectors based on metadata filters."""
        results = []
        for i, metadata in enumerate(self.metadata):
            if where is None or all(metadata.get(key) == value for key, value in where.items()):
                result = {"id": i}
                if incl is None or "metadata" in incl:
                    result["metadata"] = metadata
                if incl is None or "vector" in incl:
                    result["vector"] = self.vectors[i].tolist()
                results.append(result)
        return results
        
    def getTotalFaces(self) -> int:
      return len(self.metadata)
      
    def get_vector(self, vector_id: int) -> ndarray:
        if 0 <= vector_id < len(self.vectors):
            return self.vectors[vector_id]
        return None

    def delete_vector(self, vector_id: int) -> None:
        """Delete a vector and its metadata while maintaining correct indices."""
        if 0 <= vector_id < len(self.vectors):
            self.vectors = np.delete(self.vectors, vector_id, axis=0)
            del self.metadata[vector_id]

            self.save_vectors()
            self.save_metadata()


#db = VectorDatabase(data_dir="vector_db", collection="CppFaces",dim=512)
