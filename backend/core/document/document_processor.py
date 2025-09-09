"""
Document Processing Pipeline for PDF Analysis and RAG
Handles PDF text extraction, chunking, embedding generation, and search
"""
import os
import pickle
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Embedding and search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


class DocumentProcessor:
    """
    Handles PDF document processing, text extraction, chunking, and semantic search
    """
    
    def __init__(self, documents_dir: str = "dataset"):
        # If running from backend directory, adjust path to parent
        if Path.cwd().name == "backend":
            self.documents_dir = Path("..") / documents_dir
        else:
            self.documents_dir = Path(documents_dir)
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.model = None
        self.chunk_metadata = []
        
        # Initialize components if available
        if EMBEDDING_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # PDF files to process
        self.pdf_files = [
            "EBA_ECB 2024 Report on Payment Fraud.pdf",
            "Understanding Credit Card Frauds.pdf"
        ]
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF document using multiple methods
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries not available. Install PyPDF2 and pdfplumber.")
        
        full_path = self.documents_dir / pdf_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"PDF file not found: {full_path}")
        
        text = ""
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(full_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(full_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"PyPDF2 also failed: {e2}")
                raise e2
        
        return text.strip()
    
    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """
        Split document into overlapping chunks for better search
        
        Args:
            text: Document text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start + chunk_size // 2:  # Only if we find a good break point
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'length': len(chunk_text)
                })
            
            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Numpy array of embeddings
        """
        if not EMBEDDING_AVAILABLE:
            raise ImportError("Embedding libraries not available. Install sentence-transformers and faiss-cpu.")
        
        if not self.model:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        return embeddings
    
    def build_search_index(self, chunks: List[Dict], embeddings: np.ndarray) -> 'Any':
        """
        Build FAISS index for fast similarity search
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
            
        Returns:
            Search index for similarity search
        """
        if not EMBEDDING_AVAILABLE:
            raise ImportError("FAISS not available. Install faiss-cpu.")
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index
    
    def process_documents(self) -> bool:
        """
        Process all PDF documents and build search index
        
        Returns:
            True if successful, False otherwise
        """
        if not PDF_AVAILABLE or not EMBEDDING_AVAILABLE:
            print("Required libraries not available for document processing")
            return False
        
        all_chunks = []
        
        for pdf_file in self.pdf_files:
            try:
                print(f"Processing {pdf_file}...")
                text = self.extract_pdf_text(pdf_file)
                chunks = self.chunk_document(text)
                
                # Add metadata to chunks
                for chunk in chunks:
                    chunk['source'] = pdf_file
                    chunk['type'] = 'pdf'
                
                all_chunks.extend(chunks)
                print(f"Extracted {len(chunks)} chunks from {pdf_file}")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue
        
        if not all_chunks:
            print("No chunks extracted from documents")
            return False
        
        # Create embeddings and search index
        print("Creating embeddings...")
        self.embeddings = self.create_embeddings(all_chunks)
        self.index = self.build_search_index(all_chunks, self.embeddings)
        self.chunks = all_chunks
        self.chunk_metadata = all_chunks
        
        print(f"Successfully processed {len(all_chunks)} chunks from {len(self.pdf_files)} documents")
        return True
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant document chunks using semantic similarity
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not self.index or not self.model:
            raise RuntimeError("Document index not built. Call process_documents() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(score)
                chunk['rank'] = i + 1
                results.append(chunk)
        
        return results
    
    def search_fraud_methods(self, query: str = "fraud methods") -> List[Dict]:
        """
        Search specifically for fraud methods information
        
        Args:
            query: Search query (defaults to fraud methods)
            
        Returns:
            List of relevant chunks about fraud methods
        """
        # Enhanced query for fraud methods
        enhanced_query = f"{query} credit card fraud techniques methods how fraud is committed"
        return self.search_documents(enhanced_query, k=8)
    
    def search_system_components(self, query: str = "fraud detection system components") -> List[Dict]:
        """
        Search specifically for fraud detection system components
        
        Args:
            query: Search query (defaults to system components)
            
        Returns:
            List of relevant chunks about system components
        """
        # Enhanced query for system components
        enhanced_query = f"{query} fraud detection system architecture components effective"
        return self.search_documents(enhanced_query, k=8)
    
    def save_index(self, filepath: str) -> bool:
        """
        Save the search index and metadata to disk
        
        Args:
            filepath: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save metadata
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'chunk_metadata': self.chunk_metadata,
                    'pdf_files': self.pdf_files
                }, f)
            
            print(f"Index saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """
        Load the search index and metadata from disk
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load metadata
            with open(f"{filepath}.metadata", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_metadata = data['chunk_metadata']
                self.pdf_files = data['pdf_files']
            
            print(f"Index loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_document_summary(self) -> Dict:
        """
        Get summary of processed documents
        
        Returns:
            Dictionary with document processing summary
        """
        return {
            'total_chunks': len(self.chunks),
            'pdf_files_processed': len(self.pdf_files),
            'pdf_files': self.pdf_files,
            'index_built': self.index is not None,
            'embeddings_created': self.embeddings is not None
        }
