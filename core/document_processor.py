"""
Document Processor - Load and chunk documents
==============================================
Handles PDFs, CSVs, TXT files with extensible loader system.

Author: [Your Name]
Date: January 2026
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Callable
from datetime import datetime

from pypdf import PdfReader
import pandas as pd
from tqdm import tqdm

from core.interfaces import Document


class DocumentProcessor:
    """
    Extensible document processor.
    Add new document types by registering loaders.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Registry of document loaders
        self.loaders: Dict[str, Callable] = {
            '.pdf': self._load_pdf,
            '.csv': self._load_csv,
            '.txt': self._load_txt,
        }
        
        print(f"DocumentProcessor initialized:")
        print(f"  - Chunk size: {chunk_size} tokens")
        print(f"  - Overlap: {chunk_overlap} tokens")
        print(f"  - Supported formats: {list(self.loaders.keys())}")
    
    def register_loader(self, extension: str, loader_func: Callable):
        """
        Register custom document loader.
        
        Example:
            def load_docx(file_path):
                # Your custom loader
                return text, metadata
            
            processor.register_loader('.docx', load_docx)
        """
        self.loaders[extension] = loader_func
        print(f"✓ Registered loader for {extension}")
    
    def load_documents(self, folder_path: str) -> List[Document]:
        """
        Load all documents from a folder.
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            List of Document objects (chunked)
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        all_docs = []
        
        # Find all supported files
        files = []
        for ext in self.loaders.keys():
            files.extend(list(folder.rglob(f"*{ext}")))
        
        print(f"\nFound {len(files)} documents to process...")
        
        # Process each file
        for file_path in tqdm(files, desc="Loading documents"):
            try:
                ext = file_path.suffix.lower()
                if ext in self.loaders:
                    text, metadata = self.loaders[ext](str(file_path))
                    
                    # Chunk the document
                    chunks = self._chunk_text(text, str(file_path.name), metadata)
                    all_docs.extend(chunks)
                    
            except Exception as e:
                print(f"⚠ Error processing {file_path.name}: {e}")
                continue
        
        print(f"✓ Loaded {len(all_docs)} total chunks from {len(files)} documents")
        return all_docs
    
    def _load_pdf(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Load PDF file"""
        reader = PdfReader(file_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        metadata = {
            'type': 'pdf',
            'pages': len(reader.pages),
            'filename': Path(file_path).name
        }
        
        return text, metadata
    
    def _load_csv(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Load CSV file"""
        df = pd.read_csv(file_path)
        
        # Convert to readable text format
        text = f"Data from {Path(file_path).name}:\n\n"
        text += df.to_string()
        
        metadata = {
            'type': 'csv',
            'rows': len(df),
            'columns': list(df.columns),
            'filename': Path(file_path).name
        }
        
        return text, metadata
    
    def _load_txt(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Load text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        metadata = {
            'type': 'txt',
            'filename': Path(file_path).name
        }
        
        return text, metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common pattern)
        text = re.sub(r'Page \d+', '', text)
        
        # Remove extra newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, source: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split text into overlapping chunks.
        
        Uses simple sentence-based splitting with overlap.
        """
        # Clean text first
        text = self._clean_text(text)
        
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                chunks.append(Document(
                    text=chunk_text,
                    source=source,
                    chunk_id=f"{source}_chunk_{chunk_idx}",
                    metadata={
                        **metadata,
                        'chunk_index': chunk_idx,
                        'word_count': current_length,
                        'processed_date': datetime.now().isoformat()
                    }
                ))
                
                # Keep overlap: last few sentences
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences
                current_length = sum(len(s.split()) for s in current_chunk)
                chunk_idx += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Document(
                text=chunk_text,
                source=source,
                chunk_id=f"{source}_chunk_{chunk_idx}",
                metadata={
                    **metadata,
                    'chunk_index': chunk_idx,
                    'word_count': current_length,
                    'processed_date': datetime.now().isoformat()
                }
            ))
        
        return chunks
    
    def save_chunks(self, documents: List[Document], output_path: str):
        """Save processed chunks to JSON"""
        output = [doc.to_dict() for doc in documents]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(documents)} chunks to {output_path}")
    
    def load_chunks(self, input_path: str) -> List[Document]:
        """Load previously saved chunks"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = [
            Document(
                text=d['text'],
                source=d['source'],
                chunk_id=d['chunk_id'],
                metadata=d['metadata'],
                score=d.get('score')
            )
            for d in data
        ]
        
        print(f"✓ Loaded {len(documents)} chunks from {input_path}")
        return documents


# Quick test function
def test_processor():
    """Test the document processor"""
    print("\n" + "=" * 60)
    print("Testing Document Processor")
    print("=" * 60)
    
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    # Test with a sample text
    sample_text = """
    Copper prices reached $4.50 per pound in 2024. This represents a 15% increase 
    from the previous year. The rise is attributed to growing demand from the 
    electric vehicle sector. Each EV requires approximately 83kg of copper, which 
    is four times more than traditional vehicles. Industry analysts predict continued 
    growth in copper demand through 2026.
    
    However, supply constraints remain a concern. Major copper mines in Chile and 
    Peru have reported production declines of 2% in 2024. This creates a projected 
    supply deficit of 400,000 tonnes by 2026. Market volatility has increased, with 
    prices fluctuating between $3.80 and $4.70 per pound in the last quarter.
    """
    
    # Create test document
    test_doc = Document(
        text=sample_text,
        source="test_document.txt",
        chunk_id="test_000",
        metadata={'type': 'test'}
    )
    
    # Test cleaning
    cleaned = processor._clean_text(sample_text)
    print(f"\n✓ Text cleaning works: {len(cleaned)} characters")
    
    # Test chunking
    chunks = processor._chunk_text(sample_text, "test.txt", {'type': 'test'})
    print(f"✓ Created {len(chunks)} chunks from sample text")
    
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i}: {chunk.chunk_id}")
        print(f"  Length: {len(chunk.text.split())} words")
        print(f"  Preview: {chunk.text[:100]}...")
    
    print("\n" + "=" * 60)
    print("Document Processor Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_processor()
