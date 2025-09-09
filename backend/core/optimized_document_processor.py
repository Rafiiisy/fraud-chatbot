"""
Optimized Document Processing Pipeline for EBA_ECB 2024 Report
Chapter-aware chunking and hierarchical search for structured PDF documents
"""
import os
import pickle
import re
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


class OptimizedDocumentProcessor:
    """
    Chapter-aware document processor optimized for EBA_ECB 2024 Report structure
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
        self.chapter_index = None
        self.chunk_metadata = []
        
        # EBA Report specific chapter structure
        self.eba_chapters = {
            "executive_summary": {"title": "Executive Summary", "page_range": (5, 6)},
            "introduction": {"title": "1. Introduction", "page_range": (7, 9)},
            "fraud_levels": {"title": "2. Levels of payment fraud", "page_range": (10, 12)},
            "fraud_types": {"title": "3. Main fraud types", "page_range": (13, 16)},
            "sca_role": {"title": "4. The role of strong customer authentication (SCA)", "page_range": (17, 24)},
            "fraud_losses": {"title": "5. Losses due to fraud", "page_range": (25, 26)},
            "geographical_dimension": {"title": "6. The geographical dimension of fraud", "page_range": (27, 28)},
            "country_perspective": {"title": "7. A country-by-country and regional perspective on fraud", "page_range": (29, 32)},
            "methodology": {"title": "Annex: Reporting Methodology", "page_range": (33, 35)}
        }
        
        # Initialize components if available
        if EMBEDDING_AVAILABLE:
            # Use specialized model for financial content
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # PDF file to process
        self.pdf_file = "EBA_ECB 2024 Report on Payment Fraud.pdf"
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with better formatting preservation"""
        if not PDF_AVAILABLE:
            raise RuntimeError("PDF processing libraries not available")
        
        full_path = self.documents_dir / pdf_path
        if not full_path.exists():
            raise FileNotFoundError(f"PDF file not found: {full_path}")
        
        text = ""
        
        try:
            # Try pdfplumber first for better text extraction
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
    
    def extract_chapters_from_text(self, text: str) -> List[Dict]:
        """Extract chapters based on EBA report structure"""
        chapters = []
        lines = text.split('\n')
        
        current_chapter = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a chapter header
            chapter_found = self._identify_chapter_header(line)
            
            if chapter_found:
                # Save previous chapter if exists
                if current_chapter and current_content:
                    current_chapter['content'] = '\n'.join(current_content)
                    current_chapter['content_length'] = len(current_chapter['content'])
                    chapters.append(current_chapter)
                
                # Start new chapter
                current_chapter = {
                    'id': chapter_found['id'],
                    'title': chapter_found['title'],
                    'start_line': i,
                    'content': [],
                    'subsections': [],
                    'metadata': {
                        'chapter_id': chapter_found['id'],
                        'chapter_title': chapter_found['title'],
                        'document': 'EBA_ECB_2024_Report'
                    }
                }
                current_content = []
            else:
                # Add content to current chapter
                if current_chapter:
                    current_content.append(line)
        
        # Add the last chapter
        if current_chapter and current_content:
            current_chapter['content'] = '\n'.join(current_content)
            current_chapter['content_length'] = len(current_chapter['content'])
            chapters.append(current_chapter)
        
        return chapters
    
    def _identify_chapter_header(self, line: str) -> Optional[Dict]:
        """Identify if a line is a chapter header"""
        line_lower = line.lower()
        
        # Check against known chapter patterns
        for chapter_id, chapter_info in self.eba_chapters.items():
            title_lower = chapter_info['title'].lower()
            
            # Exact match
            if title_lower in line_lower:
                return {
                    'id': chapter_id,
                    'title': chapter_info['title'],
                    'confidence': 1.0
                }
            
            # Partial match for numbered chapters
            if chapter_id in ['introduction', 'fraud_levels', 'fraud_types', 'sca_role', 'fraud_losses', 'geographical_dimension', 'country_perspective']:
                chapter_num = chapter_id.split('_')[0] if '_' in chapter_id else chapter_id
                if f"{chapter_num}." in line_lower and len(line) > 10:
                    return {
                        'id': chapter_id,
                        'title': line,
                        'confidence': 0.8
                    }
        
        # Check for executive summary
        if 'executive summary' in line_lower and len(line) < 50:
            return {
                'id': 'executive_summary',
                'title': 'Executive Summary',
                'confidence': 0.9
            }
        
        # Check for methodology/annex
        if ('methodology' in line_lower or 'annex' in line_lower) and len(line) < 50:
            return {
                'id': 'methodology',
                'title': 'Annex: Reporting Methodology',
                'confidence': 0.9
            }
        
        return None
    
    def create_hierarchical_chunks(self, chapters: List[Dict]) -> List[Dict]:
        """Create hierarchical chunks that respect chapter boundaries"""
        chunks = []
        
        for chapter in chapters:
            # Chapter summary chunk (first 3-5 sentences)
            chapter_summary = self._extract_chapter_summary(chapter['content'])
            if chapter_summary:
                summary_chunk = {
                    'text': chapter_summary,
                    'type': 'chapter_summary',
                    'chapter_id': chapter['id'],
                    'chapter_title': chapter['title'],
                    'level': 'chapter',
                    'metadata': {
                        'chapter_id': chapter['id'],
                        'chapter_title': chapter['title'],
                        'chunk_type': 'summary',
                        'document': 'EBA_ECB_2024_Report',
                        'full_content': chapter['content']
                    }
                }
                chunks.append(summary_chunk)
            
            # Paragraph-level chunks within chapter
            paragraphs = self._split_into_paragraphs(chapter['content'])
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 100:  # Only chunk substantial paragraphs
                    para_chunk = {
                        'text': paragraph.strip(),
                        'type': 'paragraph',
                        'chapter_id': chapter['id'],
                        'chapter_title': chapter['title'],
                        'level': 'paragraph',
                        'paragraph_index': i,
                        'metadata': {
                            'chapter_id': chapter['id'],
                            'chapter_title': chapter['title'],
                            'paragraph_index': i,
                            'chunk_type': 'paragraph',
                            'document': 'EBA_ECB_2024_Report'
                        }
                    }
                    chunks.append(para_chunk)
        
        return chunks
    
    def _extract_chapter_summary(self, content: str) -> str:
        """Extract first 3-5 sentences as chapter summary"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Take first 3-5 sentences, but ensure we don't exceed 500 characters
        summary_sentences = []
        total_length = 0
        
        for sentence in sentences[:5]:
            if total_length + len(sentence) > 500:
                break
            summary_sentences.append(sentence)
            total_length += len(sentence)
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else ""
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs, respecting structure"""
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Further split very long paragraphs
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > 2000:  # Split very long paragraphs
                # Try to split at sentence boundaries
                sentences = re.split(r'[.!?]+', para)
                current_para = ""
                for sentence in sentences:
                    if len(current_para) + len(sentence) > 1000:
                        if current_para.strip():
                            final_paragraphs.append(current_para.strip())
                        current_para = sentence
                    else:
                        current_para += sentence + ". "
                if current_para.strip():
                    final_paragraphs.append(current_para.strip())
            else:
                final_paragraphs.append(para.strip())
        
        return [p for p in final_paragraphs if len(p) > 50]
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for chunks using specialized model"""
        if not self.model:
            raise RuntimeError("Embedding model not available")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize embeddings for better similarity search
        faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def build_search_index(self, chunks: List[Dict], embeddings: np.ndarray) -> 'Any':
        """Build FAISS index for fast similarity search"""
        if not EMBEDDING_AVAILABLE:
            return None
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        return index
    
    def build_chapter_index(self, chunks: List[Dict]) -> Dict[str, List[int]]:
        """Build chapter-based index for filtering"""
        chapter_index = {}
        
        for i, chunk in enumerate(chunks):
            chapter_id = chunk['chapter_id']
            if chapter_id not in chapter_index:
                chapter_index[chapter_id] = []
            chapter_index[chapter_id].append(i)
        
        return chapter_index
    
    def process_documents(self) -> bool:
        """Process EBA report and build search indices"""
        if not PDF_AVAILABLE or not EMBEDDING_AVAILABLE:
            print("Required libraries not available for document processing")
            return False
        
        try:
            print(f"Processing {self.pdf_file}...")
            
            # Extract text
            text = self.extract_text_from_pdf(self.pdf_file)
            if not text.strip():
                print("No text extracted from document")
                return False
            
            # Extract chapters
            chapters = self.extract_chapters_from_text(text)
            print(f"Extracted {len(chapters)} chapters")
            
            # Create hierarchical chunks
            chunks = self.create_hierarchical_chunks(chapters)
            print(f"Created {len(chunks)} hierarchical chunks")
            
            # Create embeddings and search index
            print("Creating embeddings...")
            self.embeddings = self.create_embeddings(chunks)
            self.index = self.build_search_index(chunks, self.embeddings)
            self.chapter_index = self.build_chapter_index(chunks)
            self.chunks = chunks
            self.chunk_metadata = chunks
            
            print(f"Successfully processed {len(chunks)} chunks from {len(chapters)} chapters")
            return True
            
        except Exception as e:
            print(f"Error processing documents: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5, chapter_filter: Optional[str] = None) -> List[Dict]:
        """Search documents with optional chapter filtering"""
        if not self.index or not self.model:
            raise RuntimeError("Document index not built. Call process_documents() first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Determine search scope
        if chapter_filter and chapter_filter in self.chapter_index:
            # Search only within specific chapter
            chapter_indices = self.chapter_index[chapter_filter]
            search_k = min(k * 2, len(chapter_indices))  # Search more to get better results
        else:
            # Search all chunks
            chapter_indices = None
            search_k = k * 3  # Search more to get better results
        
        # Search for similar chunks
        if chapter_indices:
            # Create sub-index for chapter
            chapter_embeddings = self.embeddings[chapter_indices]
            sub_index = faiss.IndexFlatIP(chapter_embeddings.shape[1])
            sub_index.add(chapter_embeddings.astype('float32'))
            
            scores, local_indices = sub_index.search(query_embedding, search_k)
            # Convert local indices back to global indices
            indices = [chapter_indices[i] for i in local_indices[0]]
        else:
            scores, indices = self.index.search(query_embedding, search_k)
            indices = indices[0]
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices)):
            if i >= k:
                break
                
            chunk = self.chunks[idx]
            result = {
                'text': chunk['text'],
                'score': float(score),
                'chapter_id': chunk['chapter_id'],
                'chapter_title': chunk['chapter_title'],
                'chunk_type': chunk['type'],
                'metadata': chunk['metadata']
            }
            results.append(result)
        
        return results
    
    def get_chapter_summary(self, chapter_id: str) -> Optional[Dict]:
        """Get summary of a specific chapter"""
        if not self.chunks:
            return None
        
        for chunk in self.chunks:
            if (chunk['chapter_id'] == chapter_id and 
                chunk['type'] == 'chapter_summary'):
                return {
                    'chapter_id': chapter_id,
                    'title': chunk['chapter_title'],
                    'summary': chunk['text'],
                    'metadata': chunk['metadata']
                }
        
        return None
    
    def get_available_chapters(self) -> List[Dict]:
        """Get list of available chapters"""
        if not self.chunks:
            return []
        
        chapters = {}
        for chunk in self.chunks:
            if chunk['type'] == 'chapter_summary':
                chapters[chunk['chapter_id']] = {
                    'id': chunk['chapter_id'],
                    'title': chunk['chapter_title'],
                    'summary': chunk['text']
                }
        
        return list(chapters.values())
    
    def save_index(self, index_path: str):
        """Save the search index to disk"""
        if not self.index:
            return
        
        index_data = {
            'index': self.index,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'chapter_index': self.chapter_index,
            'chunk_metadata': self.chunk_metadata
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, index_path: str) -> bool:
        """Load the search index from disk"""
        try:
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.index = index_data['index']
            self.chunks = index_data['chunks']
            self.embeddings = index_data['embeddings']
            self.chapter_index = index_data['chapter_index']
            self.chunk_metadata = index_data['chunk_metadata']
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
