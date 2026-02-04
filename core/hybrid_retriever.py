"""
Hybrid Retriever - Combines FAISS + BM25 with RRF
==================================================
Reciprocal Rank Fusion merges semantic and keyword search.

Author: [Your Name]
Date: January 2026
"""

from typing import List, Dict
from collections import defaultdict

from core.interfaces import BaseRetriever, Document


class HybridRetriever:
    """
    Combines multiple retrievers using Reciprocal Rank Fusion (RRF).
    
    RRF Formula: score(doc) = sum(1 / (k + rank_i))
    where k=60 is a constant, rank_i is position in each retriever's results.
    """
    
    def __init__(self, retrievers: List[BaseRetriever], weights: Dict[str, float] = None):
        """
        Initialize hybrid retriever.
        
        Args:
            retrievers: List of retriever instances
            weights: Optional weights for each retriever (default: equal weight)
        """
        self.retrievers = {r.get_name(): r for r in retrievers}
        
        # Default to equal weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in self.retrievers.keys()}
        
        self.weights = weights
        self.rrf_k = 60  # RRF constant
        
        print(f"✓ HybridRetriever initialized with {len(self.retrievers)} retrievers:")
        for name, weight in self.weights.items():
            print(f"  - {name}: weight={weight}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            List of documents ranked by combined scores
        """
        # Get results from each retriever
        all_results = {}
        
        for name, retriever in self.retrievers.items():
            try:
                # Get more results than needed for better fusion
                results = retriever.retrieve(query, top_k=top_k * 2)
                all_results[name] = results
                print(f"  {name}: {len(results)} results")
            except Exception as e:
                print(f"⚠ {name} retrieval failed: {e}")
                all_results[name] = []
        
        # Apply Reciprocal Rank Fusion
        fused_scores = self._reciprocal_rank_fusion(all_results)
        
        # Sort by fused score and return top-k
        ranked_docs = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build final document list with combined scores
        results = []
        for chunk_id, score in ranked_docs:
            # Find original document from any retriever
            doc = self._find_document(chunk_id, all_results)
            if doc:
                # Update with fused score
                doc.score = score
                results.append(doc)
        
        return results
    
    def _reciprocal_rank_fusion(self, 
                                all_results: Dict[str, List[Document]]) -> Dict[str, float]:
        """
        Apply RRF algorithm to merge rankings.
        
        Args:
            all_results: Dict mapping retriever name to its results
            
        Returns:
            Dict mapping chunk_id to fused score
        """
        fused_scores = defaultdict(float)
        
        for retriever_name, documents in all_results.items():
            weight = self.weights.get(retriever_name, 1.0)
            
            # Calculate RRF score for each document
            for rank, doc in enumerate(documents, start=1):
                rrf_score = weight * (1.0 / (self.rrf_k + rank))
                fused_scores[doc.chunk_id] += rrf_score
        
        return dict(fused_scores)
    
    def _find_document(self, chunk_id: str, 
                      all_results: Dict[str, List[Document]]) -> Document:
        """Find document by chunk_id from any retriever's results"""
        for documents in all_results.values():
            for doc in documents:
                if doc.chunk_id == chunk_id:
                    return doc
        return None
    
    def add_documents(self, documents: List[Document]):
        """Add documents to all retrievers"""
        print(f"\nAdding {len(documents)} documents to all retrievers...")
        
        for name, retriever in self.retrievers.items():
            try:
                retriever.add_documents(documents)
            except Exception as e:
                print(f"⚠ Failed to add documents to {name}: {e}")
    
    def get_retriever(self, name: str) -> BaseRetriever:
        """Get a specific retriever by name"""
        return self.retrievers.get(name)
    
    def update_weights(self, weights: Dict[str, float]):
        """Update retriever weights"""
        self.weights.update(weights)
        print(f"✓ Updated weights: {self.weights}")


# Test function
def test_hybrid():
    """Test hybrid retriever"""
    print("\n" + "=" * 60)
    print("Testing Hybrid Retriever")
    print("=" * 60)
    
    from retrievers.faiss_retriever import FAISSRetriever
    from retrievers.bm25_retriever import BM25Retriever
    
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
            text="Copper supply constraints in Chilean mines affect production.",
            source="doc4.txt",
            chunk_id="doc4_chunk_0",
            metadata={'type': 'test'}
        ),
        Document(
            text="Renewable energy projects require significant copper quantities.",
            source="doc5.txt",
            chunk_id="doc5_chunk_0",
            metadata={'type': 'test'}
        ),
    ]
    
    # Initialize retrievers
    print("\n[1] Initializing retrievers...")
    faiss = FAISSRetriever(device="cuda")
    bm25 = BM25Retriever()
    
    # Create hybrid retriever
    print("\n[2] Creating hybrid retriever...")
    hybrid = HybridRetriever(
        retrievers=[faiss, bm25],
        weights={'faiss': 0.6, 'bm25': 0.4}
    )
    
    # Add documents
    print("\n[3] Adding documents...")
    hybrid.add_documents(sample_docs)
    
    # Test query
    query = "How much copper do electric vehicles need?"
    print(f"\n[4] Testing query: '{query}'")
    print("\nHybrid Search Results:")
    
    results = hybrid.retrieve(query, top_k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Fused Score: {doc.score:.4f}")
        print(f"   Source: {doc.source}")
        print(f"   Text: {doc.text}")
    
    print("\n" + "=" * 60)
    print("Comparison with Individual Retrievers:")
    print("=" * 60)
    
    # Compare with individual retrievers
    print("\nFAISS Only:")
    faiss_results = faiss.retrieve(query, top_k=3)
    for i, doc in enumerate(faiss_results, 1):
        print(f"{i}. {doc.source} (score: {doc.score:.3f})")
    
    print("\nBM25 Only:")
    bm25_results = bm25.retrieve(query, top_k=3)
    for i, doc in enumerate(bm25_results, 1):
        print(f"{i}. {doc.source} (score: {doc.score:.3f})")
    
    print("\nHybrid (FAISS 60% + BM25 40%):")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.source} (fused score: {doc.score:.4f})")
    
    print("\n" + "=" * 60)
    print("✓ Hybrid Retriever combines best of both methods!")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid()
