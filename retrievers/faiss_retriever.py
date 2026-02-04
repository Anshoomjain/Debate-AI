"""
FAISS Retriever - Semantic search using vector embeddings
==========================================================
GPU-accelerated similarity search.

Author: [Your Name]
Date: January 2026
"""

import numpy as np
import faiss
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from core.interfaces import BaseRetriever, Document


class FAISSRetriever(BaseRetriever):
    """
    Semantic search using FAISS vector database.
    Converts text to embeddings and finds similar documents.
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda"):
        super().__init__(name="faiss")
        
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Load embedding model
        print(f"Loading embedding model on {device.upper()}...")
        self.encoder = SentenceTransformer(model_name, device=device)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        
        # FAISS index
        self.index = None
        self.documents: List[Document] = []
        
        print(f"✓ FAISSRetriever initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Dimension: {self.dimension}")
        print(f"  - Device: {device.upper()}")
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to FAISS index.
        
        Args:
            documents: List of Document objects to index
        """
        if not documents:
            raise ValueError("Cannot add empty document list")
        
        print(f"\nIndexing {len(documents)} documents...")
        
        # Store documents
        self.documents = documents
        
        # Extract texts
        texts = [doc.text for doc in documents]
        
        # Generate embeddings (with progress bar)
        print("Generating embeddings...")
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Create FAISS index
        # Try GPU first, fallback to CPU if not available
        try:
            if self.device == "cuda" and hasattr(faiss, 'StandardGpuResources'):
                # GPU index (requires faiss-gpu)
                res = faiss.StandardGpuResources()
                index_flat = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                print("✓ Using GPU-accelerated FAISS index")
            else:
                raise AttributeError("GPU not available")
        except (AttributeError, RuntimeError) as e:
            # Fallback to CPU
            self.index = faiss.IndexFlatL2(self.dimension)
            print("✓ Using CPU FAISS index (install faiss-gpu for GPU acceleration)")
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Indexed {self.index.ntotal} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve most similar documents to query.
        
        Args:
            query: Search query
            top_k: Number of documents to return
            
        Returns:
            List of Document objects with similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No documents indexed. Call add_documents() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert distances to similarity scores (cosine similarity)
        # FAISS L2 distance: convert to similarity [0, 1]
        # similarity = 1 / (1 + distance)
        similarities = 1 / (1 + distances[0])
        
        # Retrieve documents with scores
        results = []
        for idx, score in zip(indices[0], similarities):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # Create new document with score
                result = Document(
                    text=doc.text,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata,
                    score=float(score)
                )
                results.append(result)
        
        return results
    
    def save_index(self, path: str):
        """Save FAISS index to disk"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Check if index is on GPU
        try:
            if hasattr(faiss, 'index_gpu_to_cpu'):
                # Try to convert from GPU to CPU
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, path)
            else:
                # Already CPU or faiss-cpu version
                faiss.write_index(self.index, path)
        except (RuntimeError, AttributeError):
            # Fallback - just save as-is
            faiss.write_index(self.index, path)
        
        print(f"✓ Saved FAISS index to {path}")
    
    def load_index(self, path: str, documents: List[Document]):
        """Load FAISS index from disk"""
        # Load index
        index_cpu = faiss.read_index(path)
        
        # Try to move to GPU if available
        try:
            if self.device == "cuda" and hasattr(faiss, 'index_cpu_to_gpu'):
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
                print("✓ Loaded index on GPU")
            else:
                raise AttributeError("GPU not available")
        except (RuntimeError, AttributeError):
            # Use CPU
            self.index = index_cpu
            print("✓ Loaded index on CPU")
        
        self.documents = documents
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")


# Test function
def test_faiss():
    """Test FAISS retriever"""
    print("\n" + "=" * 60)
    print("Testing FAISS Retriever")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            text="Copper prices increased 15% due to EV demand growth.",
            source="doc1.txt",
            chunk_id="doc1_chunk_0",
            metadata={'type': 'test'}
        ),
        Document(
            text="Electric vehicles require 83kg of copper per car.",
            source="doc2.txt",
            chunk_id="doc2_chunk_0",
            metadata={'type': 'test'}
        ),
        Document(
            text="Gold prices remain stable in current market conditions.",
            source="doc3.txt",
            chunk_id="doc3_chunk_0",
            metadata={'type': 'test'}
        ),
        Document(
            text="Supply constraints in Chilean mines affect production.",
            source="doc4.txt",
            chunk_id="doc4_chunk_0",
            metadata={'type': 'test'}
        ),
    ]
    
    # Initialize retriever
    retriever = FAISSRetriever(device="cuda")
    
    # Index documents
    retriever.add_documents(sample_docs)
    
    # Test query
    query = "How much copper is needed for electric vehicles?"
    print(f"\nQuery: '{query}'")
    print("\nTop 3 Results:")
    
    results = retriever.retrieve(query, top_k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Score: {doc.score:.3f}")
        print(f"   Source: {doc.source}")
        print(f"   Text: {doc.text[:100]}...")
    
    print("\n" + "=" * 60)
    print("FAISS Retriever Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_faiss()
