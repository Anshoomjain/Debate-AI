"""
BM25 Retriever - Keyword-based search
======================================
Traditional information retrieval using BM25 algorithm.

Author: [Your Name]
Date: January 2026
"""

from typing import List
from rank_bm25 import BM25Okapi
import pickle

from core.interfaces import BaseRetriever, Document


class BM25Retriever(BaseRetriever):
    """
    Keyword search using BM25 algorithm.
    Complements semantic search by catching exact term matches.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(name="bm25")
        
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization
        
        self.bm25 = None
        self.documents: List[Document] = []
        
        print(f"✓ BM25Retriever initialized:")
        print(f"  - k1: {k1}")
        print(f"  - b: {b}")
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to BM25 index.
        
        Args:
            documents: List of Document objects to index
        """
        if not documents:
            raise ValueError("Cannot add empty document list")
        
        print(f"\nIndexing {len(documents)} documents with BM25...")
        
        # Store documents
        self.documents = documents
        
        # Tokenize documents (simple word-based tokenization)
        tokenized_docs = [self._tokenize(doc.text) for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        
        print(f"✓ Indexed {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents matching query keywords.
        
        Args:
            query: Search query
            top_k: Number of documents to return
            
        Returns:
            List of Document objects with BM25 scores
        """
        if self.bm25 is None:
            raise ValueError("No documents indexed. Call add_documents() first.")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Normalize scores to [0, 1] range
        max_score = scores.max() if scores.max() > 0 else 1.0
        
        # Retrieve documents with scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            normalized_score = scores[idx] / max_score
            
            result = Document(
                text=doc.text,
                source=doc.source,
                chunk_id=doc.chunk_id,
                metadata=doc.metadata,
                score=float(normalized_score)
            )
            results.append(result)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase and split on whitespace.
        Can be extended with stemming, lemmatization, etc.
        """
        return text.lower().split()
    
    def save_index(self, path: str):
        """Save BM25 index to disk"""
        if self.bm25 is None:
            raise ValueError("No index to save")
        
        with open(path, 'wb') as f:
            pickle.dump((self.bm25, self.documents), f)
        
        print(f"✓ Saved BM25 index to {path}")
    
    def load_index(self, path: str):
        """Load BM25 index from disk"""
        with open(path, 'rb') as f:
            self.bm25, self.documents = pickle.load(f)
        
        print(f"✓ Loaded BM25 index with {len(self.documents)} documents")


# Test function
def test_bm25():
    """Test BM25 retriever"""
    print("\n" + "=" * 60)
    print("Testing BM25 Retriever")
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
            text="Copper supply constraints in Chilean mines.",
            source="doc4.txt",
            chunk_id="doc4_chunk_0",
            metadata={'type': 'test'}
        ),
    ]
    
    # Initialize retriever
    retriever = BM25Retriever()
    
    # Index documents
    retriever.add_documents(sample_docs)
    
    # Test query
    query = "copper prices electric vehicles"
    print(f"\nQuery: '{query}'")
    print("\nTop 3 Results:")
    
    results = retriever.retrieve(query, top_k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Score: {doc.score:.3f}")
        print(f"   Source: {doc.source}")
        print(f"   Text: {doc.text}")
    
    print("\n" + "=" * 60)
    print("BM25 Retriever Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_bm25()
