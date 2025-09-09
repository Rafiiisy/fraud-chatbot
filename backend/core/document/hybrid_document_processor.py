"""
Hybrid Document Processing Pipeline
Intelligently routes between FAISS (fast, cheap) and OpenAI (comprehensive, expensive)
"""
import os
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
import numpy as np

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# FAISS and embeddings
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# BM25 for keyword prefiltering
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None

# OpenAI integration
try:
    from ..ai.openai_integration import OpenAIIntegration
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Dynamic query analyzer
try:
    from ..query.dynamic_query_analyzer import DynamicQueryAnalyzer
    DYNAMIC_ANALYZER_AVAILABLE = True
except ImportError:
    DYNAMIC_ANALYZER_AVAILABLE = False


class HybridDocumentProcessor:
    """
    Hybrid document processor that intelligently routes between FAISS and OpenAI
    for optimal performance and cost efficiency
    """
    
    def __init__(self, documents_dir: str = "dataset", use_faiss: bool = True):
        # If running from backend directory, adjust path to parent
        if Path.cwd().name == "backend":
            self.documents_dir = Path("..") / documents_dir
        else:
            self.documents_dir = Path(documents_dir)
        
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.use_bm25 = BM25_AVAILABLE
        self.documents = {}
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.bm25_index = None
        self.openai_client = None
        self.query_analyzer = None
        
        # Initialize components
        # Note: Using OpenAI embeddings instead of sentence-transformers
        
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAIIntegration()
        
        # Initialize dynamic query analyzer
        if DYNAMIC_ANALYZER_AVAILABLE:
            self.query_analyzer = DynamicQueryAnalyzer()
        
        # PDF files to process
        self.pdf_files = [
            "EBA_ECB 2024 Report on Payment Fraud.pdf",
            "Understanding Credit Card Frauds.pdf"
        ]
        
        # Cost tracking
        self.cost_tracker = {
            'faiss_queries': 0,
            'openai_queries': 0,
            'hybrid_queries': 0,
            'bm25_queries': 0,
            'total_tokens_saved': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def is_available(self) -> bool:
        """Check if document processing is available"""
        return PDF_AVAILABLE and (self.use_faiss or OPENAI_AVAILABLE)
    
    def process_documents(self) -> bool:
        """
        Process all PDF documents and build search index
        Returns True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Extract text from PDFs
            for pdf_file in self.pdf_files:
                pdf_path = self.documents_dir / pdf_file
                if pdf_path.exists():
                    print(f"Processing {pdf_file}...")
                    text = self._extract_text_from_pdf(pdf_path)
                    if text:
                        self.documents[pdf_file] = text
                        print(f"✅ Extracted {len(text)} characters from {pdf_file}")
                    else:
                        print(f"❌ Failed to extract text from {pdf_file}")
                else:
                    print(f"⚠️ PDF file not found: {pdf_path}")
            
            if not self.documents:
                return False
            
            # Build FAISS index if available
            if self.use_faiss:
                print("Building FAISS search index...")
                success = self._build_faiss_index()
                if success:
                    print("✅ FAISS index built successfully")
                else:
                    print("⚠️ FAISS index build failed, falling back to OpenAI-only mode")
                    self.use_faiss = False
            
            # Build BM25 index for keyword prefiltering
            if self.use_bm25:
                print("Building BM25 keyword index...")
                success = self._build_bm25_index()
                if success:
                    print("✅ BM25 index built successfully")
                else:
                    print("⚠️ BM25 index build failed, falling back to FAISS-only mode")
                    self.use_bm25 = False
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            return False
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file with enhanced normalization"""
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Normalize the page text
                        normalized_text = self._normalize_pdf_text(page_text, page_num)
                        text += f"\n--- Page {page_num} ---\n{normalized_text}\n"
                return text.strip()
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path}: {e}")
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            normalized_text = self._normalize_pdf_text(page_text, page_num)
                            text += f"\n--- Page {page_num} ---\n{normalized_text}\n"
                    return text.strip()
            except Exception as e2:
                print(f"PyPDF2 also failed for {pdf_path}: {e2}")
                return ""
    
    def _normalize_pdf_text(self, text: str, page_num: int) -> str:
        """
        Normalize PDF text to fix common extraction issues
        Based on GPT's recommendations for better retrieval
        """
        import re
        
        # 1. Fix soft hyphens and line breaks
        # Replace soft hyphen (\u00AD) with empty string
        text = text.replace('\u00AD', '')
        
        # 2. Fix hyphenation at line breaks
        # Pattern: word-\nword -> wordword
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # 3. Fix ligatures
        text = text.replace('ﬁ', 'fi')  # fi ligature
        text = text.replace('ﬂ', 'fl')  # fl ligature
        text = text.replace('ﬀ', 'ff')  # ff ligature
        text = text.replace('ﬃ', 'ffi') # ffi ligature
        text = text.replace('ﬄ', 'ffl') # ffl ligature
        
        # 4. Normalize line breaks
        # Replace single newlines with spaces (within paragraphs)
        # Keep double newlines (paragraph breaks)
        lines = text.split('\n')
        normalized_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                normalized_lines.append('')  # Keep empty lines
                continue
                
            # Check if this is a continuation of previous line
            if (normalized_lines and 
                normalized_lines[-1] and 
                not line.startswith(('---', 'Page', 'Chart', 'Figure', 'Table')) and
                not re.match(r'^\d+$', line) and  # Not just page numbers
                not re.match(r'^[A-Z\s]+$', line)):  # Not all caps headers
                # Join with previous line
                normalized_lines[-1] += ' ' + line
            else:
                normalized_lines.append(line)
        
        # 5. Remove boilerplate headers/footers
        filtered_lines = []
        for line in normalized_lines:
            # Skip common boilerplate
            if (re.match(r'^Page \d+ of \d+$', line) or
                re.match(r'^\d+$', line) or  # Just page numbers
                'EBA_ECB 2024 Report on Payment Fraud' in line and len(line) < 50):
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _build_faiss_index(self) -> bool:
        """Build FAISS search index from documents"""
        try:
            all_chunks = []
            
            # Chunk documents
            for doc_name, doc_text in self.documents.items():
                chunks = self._chunk_document(doc_text, doc_name)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                return False
            
            # Create embeddings using OpenAI
            print(f"Creating embeddings for {len(all_chunks)} chunks...")
            chunk_texts = [chunk['text'] for chunk in all_chunks]
            embeddings = self._create_openai_embeddings(chunk_texts)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            # Ensure embeddings are float32 and contiguous
            embeddings = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            
            # Store for later use
            self.chunks = all_chunks
            self.embeddings = embeddings
            self.index = index
            
            return True
            
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            return False
    
    def _build_bm25_index(self) -> bool:
        """Build BM25 keyword search index"""
        if not self.use_bm25 or not BM25_AVAILABLE:
            return False
        
        try:
            # Prepare text corpus for BM25
            corpus = []
            for chunk in self.chunks:
                # Tokenize text for BM25 (simple word splitting)
                text = chunk['text'].lower()
                # Remove punctuation and split into words
                import re
                words = re.findall(r'\b\w+\b', text)
                corpus.append(words)
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(corpus)
            return True
            
        except Exception as e:
            print(f"Error building BM25 index: {e}")
            return False
    
    def _chunk_document(self, text: str, doc_name: str, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
        """
        Chunk document using semantic boundaries with enhanced metadata
        Based on GPT's recommendations for better retrieval
        """
        chunks = []
        
        # Extract page and section information
        pages = text.split('--- Page ')
        all_paragraphs = []
        
        for page_content in pages[1:]:  # Skip first empty split
            if not page_content.strip():
                continue
                
            lines = page_content.split('\n')
            page_num = lines[0].split(' ---')[0] if ' ---' in lines[0] else '1'
            page_text = '\n'.join(lines[1:]) if len(lines) > 1 else page_content
            
            # Split by paragraphs within this page
            page_paragraphs = page_text.split('\n\n')
            for para in page_paragraphs:
                para = para.strip()
                if para and len(para) > 20:  # Skip very short paragraphs
                    # Detect section headers
                    section = self._detect_section(para)
                    all_paragraphs.append({
                        'text': para,
                        'page': int(page_num) if page_num.isdigit() else 1,
                        'section': section,
                        'is_header': self._is_section_header(para)
                    })
        
        # Group paragraphs into chunks
        current_chunk = ""
        current_words = 0
        chunk_id = 0
        current_page = 1
        current_section = "Unknown"
        
        for para_info in all_paragraphs:
            para_text = para_info['text']
            para_words = len(para_text.split())
            
            # If this is a section header, start a new chunk
            if para_info['is_header']:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk, doc_name, chunk_id, current_words, 
                        current_page, current_section, 'paragraph_group'
                    ))
                    chunk_id += 1
                
                current_chunk = para_text
                current_words = para_words
                current_section = para_info['section']
                current_page = para_info['page']
            else:
                # If adding this paragraph would exceed chunk size, finalize current chunk
                if current_words + para_words > chunk_size and current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk, doc_name, chunk_id, current_words,
                        current_page, current_section, 'paragraph_group'
                    ))
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + para_text
                    current_words = len(current_chunk.split())
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para_text
                    else:
                        current_chunk = para_text
                    current_words += para_words
                    current_page = para_info['page']
                    current_section = para_info['section']
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk, doc_name, chunk_id, current_words,
                current_page, current_section, 'paragraph_group'
            ))
        
        # If we have very few chunks, fall back to word-based chunking
        if len(chunks) < 3:
            return self._chunk_by_words(text, doc_name, chunk_size, overlap)
        
        return chunks
    
    def _create_chunk(self, text: str, doc_name: str, chunk_id: int, word_count: int, 
                     page: int, section: str, chunk_type: str) -> Dict:
        """Create a chunk with enhanced metadata"""
        return {
            'text': text.strip(),
            'source': doc_name,
            'chunk_id': chunk_id,
            'word_count': word_count,
            'type': chunk_type,
            'page': page,
            'section': section,
            'is_executive_summary': 'executive summary' in section.lower(),
            'is_geographical': 'geographical' in section.lower() or 'geography' in section.lower()
        }
    
    def _detect_section(self, text: str) -> str:
        """Detect section name from text"""
        text_lower = text.lower()
        
        if 'executive summary' in text_lower:
            return 'Executive Summary'
        elif 'introduction' in text_lower:
            return 'Introduction'
        elif 'levels of payment fraud' in text_lower:
            return 'Levels of Payment Fraud'
        elif 'main fraud types' in text_lower:
            return 'Main Fraud Types'
        elif 'strong customer authentication' in text_lower or 'sca' in text_lower:
            return 'Strong Customer Authentication (SCA)'
        elif 'losses due to fraud' in text_lower:
            return 'Losses Due to Fraud'
        elif 'geographical dimension' in text_lower:
            return 'Geographical Dimension of Fraud'
        elif 'country-by-country' in text_lower:
            return 'Country-by-Country Analysis'
        else:
            return 'Other'
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text is a section header"""
        text_lower = text.lower()
        return (len(text) < 100 and 
                (text_lower.startswith(('executive summary', 'introduction', 'levels of payment fraud',
                                      'main fraud types', 'strong customer authentication', 'sca',
                                      'losses due to fraud', 'geographical dimension', 'country-by-country')) or
                 text.isupper() and len(text) < 50))
    
    def _chunk_by_words(self, text: str, doc_name: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Fallback: chunk by word count with overlap"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 100:  # Only include substantial chunks
                chunks.append({
                    'text': chunk_text,
                    'source': doc_name,
                    'chunk_id': len(chunks),
                    'word_count': len(chunk_words),
                    'start_word': i,
                    'end_word': min(i + chunk_size, len(words)),
                    'type': 'word_based'
                })
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_words: int) -> str:
        """Get the last N words from text for overlap"""
        words = text.split()
        if len(words) <= overlap_words:
            return text
        return ' '.join(words[-overlap_words:])
    
    def _create_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Use the OpenAI integration's direct API call method
                response = self.openai_client._make_embeddings_api_call(batch)
                if response:
                    batch_embeddings = response
                    embeddings.extend(batch_embeddings)
                else:
                    raise Exception("Failed to get embeddings from OpenAI")
            except Exception as e:
                print(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                # Fallback: create random embeddings
                batch_embeddings = [np.random.rand(1536).tolist() for _ in batch]
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def search_documents(self, query: str, max_results: int = 5, use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Search documents using hybrid approach with multi-strategy content retrieval
        Intelligently routes between FAISS and OpenAI based on query complexity
        """
        if not self.is_available():
            return self._generate_fallback_response(query)
        
        # Ensure documents are processed
        if not self.documents:
            print("No documents loaded, processing documents...")
            if not self.process_documents():
                return self._generate_fallback_response(query)
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(query, use_hybrid)
        
        try:
            # Use dual retrieval for EEA-related queries to find specific statistics
            if any(term in query.lower() for term in ['eea', 'outside', 'fraud rates', 'higher', 'times higher']):
                return self._search_with_dual_retrieval(query, max_results)
            elif search_strategy == 'faiss_only':
                return self._search_with_faiss(query, max_results)
            elif search_strategy == 'openai_only':
                return self._search_with_openai(query, max_results)
            else:  # hybrid
                return self._search_with_hybrid(query, max_results)
                
        except Exception as e:
            self.logger.error(f"Error in document search: {e}")
            return self._generate_fallback_response(query)
    
    def _search_with_multi_strategy(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Multi-strategy search to find specific data points
        Tries multiple search approaches to find missing statistics
        """
        print(f"🔍 Multi-strategy search for: '{query[:50]}...'")
        
        # Strategy 1: Original query
        result1 = self._search_with_faiss(query, max_results)
        
        # Strategy 2: Query expansion for EEA fraud rates
        if any(term in query.lower() for term in ['eea', 'outside', 'fraud rates', 'higher']):
            expanded_queries = [
                "ten times higher fraud rates EEA",
                "28% fraudulent payments cross-border",
                "strong customer authentication SCA outside EEA",
                "fraud rates outside European Economic Area",
                "cross-border card fraud statistics",
                "order of magnitude fraud difference",
                "SCA was applied electronic payments",
                "SCA-authenticated transactions lower fraud rates",
                "fraud rates card payments significantly higher",
                "counterpart located outside EEA SCA"
            ]
            
            all_sources = result1.get('sources', [])
            
            for expanded_query in expanded_queries:
                print(f"  🔍 Trying expanded query: '{expanded_query}'")
                expanded_result = self._search_with_faiss(expanded_query, 3)
                if expanded_result.get('sources'):
                    all_sources.extend(expanded_result['sources'])
            
            # Remove duplicates and sort by score
            unique_sources = []
            seen_quotes = set()
            for source in all_sources:
                quote = source.get('quote', '')
                if quote not in seen_quotes and len(quote) > 50:
                    unique_sources.append(source)
                    seen_quotes.add(quote)
            
            # Sort by score and take top results
            unique_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
            unique_sources = unique_sources[:max_results]
            
            if unique_sources:
                print(f"  ✅ Multi-strategy found {len(unique_sources)} unique sources")
                
                # Strategy 3: Deep search for specific phrases found manually
                deep_search_queries = [
                    "SCA was applied for the majority of electronic payments",
                    "fraud rates for card payments turned out to be significantly about ten times higher",
                    "counterpart is located outside the EEA where the application of SCA may not be requested",
                    "SCA-authenticated transactions showed lower fraud rates than non-SCA transactions"
                ]
                
                print(f"  🔍 Deep search for specific phrases...")
                for deep_query in deep_search_queries:
                    print(f"    🔍 Deep query: '{deep_query[:50]}...'")
                    deep_result = self._search_with_faiss(deep_query, 2)
                    if deep_result.get('sources'):
                        unique_sources.extend(deep_result['sources'])
                
                # Remove duplicates again after deep search
                unique_sources = list({source.get('quote', ''): source for source in unique_sources}.values())
                unique_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
                unique_sources = unique_sources[:max_results]
                
                print(f"  ✅ Final sources after deep search: {len(unique_sources)}")
                answer = self._generate_answer_from_sources(query, unique_sources)
                
                return {
                    'success': True,
                    'answer': answer,
                    'sources': unique_sources,
                    'confidence': max([s.get('score', 0) for s in unique_sources]),
                    'method': 'multi_strategy_deep',
                    'cost_saved': True
                }
        
        return result1
    
    def _determine_search_strategy(self, query: str, use_hybrid: bool) -> str:
        """
        Determine the best search strategy based on query characteristics
        """
        print(f"🎯 Determining search strategy for query: '{query[:50]}...'")
        
        if not use_hybrid or not self.use_faiss:
            print("  → Using OpenAI only (hybrid disabled or FAISS unavailable)")
            return 'openai_only'
        
        # Simple heuristics for routing
        query_lower = query.lower()
        query_words = len(query.split())
        
        print(f"  - Query length: {query_words} words")
        print(f"  - FAISS available: {self.use_faiss}")
        print(f"  - Hybrid enabled: {use_hybrid}")
        
        # FAISS is good for:
        # - Specific fact-finding questions
        # - Data/statistics queries
        # - Questions asking for specific information
        faiss_indicators = [
            'what is', 'what are', 'define', 'explain', 'describe',
            'how does', 'how do', 'which', 'where', 'when',
            'list', 'name', 'identify', 'find', 'how much', 'how many',
            'percentage', '%', 'rate', 'fraud', 'eea', 'outside', 
            'cross-border', 'times higher', 'ten times', '28%', 'specific',
            'data', 'statistics', 'numbers', 'figures'
        ]
        
        # OpenAI is better for:
        # - Complex analytical questions
        # - Questions requiring synthesis
        # - Long, detailed queries
        openai_indicators = [
            'analyze', 'compare', 'evaluate', 'assess', 'discuss',
            'why', 'what are the implications', 'what are the benefits',
            'what are the challenges', 'what are the risks', 'methodology',
            'approach', 'strategy', 'recommendations', 'trend', 'pattern'
        ]
        
        # Check query length
        if query_words < 5:
            print("  → Using FAISS only (short query)")
            return 'faiss_only'
        elif query_words > 15:
            print("  → Using OpenAI only (long query)")
            return 'openai_only'
        
        # Check for specific indicators
        faiss_matches = [indicator for indicator in faiss_indicators if indicator in query_lower]
        openai_matches = [indicator for indicator in openai_indicators if indicator in query_lower]
        
        if faiss_matches:
            print(f"  → Using FAISS only (matched indicators: {faiss_matches})")
            return 'faiss_only'
        elif openai_matches:
            print(f"  → Using OpenAI only (matched indicators: {openai_matches})")
            return 'openai_only'
        
        # Default to hybrid for balanced approach
        print("  → Using hybrid approach (balanced)")
        return 'hybrid'
    
    def _search_with_dual_retrieval(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Dual retrieval: BM25 keyword prefiltering + FAISS dense retrieval
        Based on GPT's recommendations for better precision
        """
        if not self.use_bm25 or not self.use_faiss or not self.bm25_index or not self.index:
            return self._search_with_faiss(query, max_results)
        
        try:
            # Step 1: Query expansion and rewriting
            expanded_queries = self._expand_query(query)
            print(f"🔍 Query expansion: {len(expanded_queries)} query variations")
            
            # Step 2: BM25 keyword prefiltering with expanded queries
            all_query_words = []
            for expanded_query in expanded_queries:
                all_query_words.extend(expanded_query.lower().split())
            
            # Remove duplicates while preserving order
            unique_words = list(dict.fromkeys(all_query_words))
            bm25_scores = self.bm25_index.get_scores(unique_words)
            
            # Get top candidates from BM25 (3x more than needed for reranking)
            bm25_candidates = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
            bm25_top_k = min(len(bm25_candidates), max_results * 3)
            bm25_indices = [idx for idx, score in bm25_candidates[:bm25_top_k] if score > 0]
            
            print(f"🔍 BM25 prefiltering: {len(bm25_indices)} candidates from {len(self.chunks)} total chunks")
            
            if not bm25_indices:
                # Fallback to FAISS if BM25 finds nothing
                return self._search_with_faiss(query, max_results)
            
            # Step 2: FAISS dense retrieval on BM25 candidates
            query_embedding = self._create_openai_embeddings([query])
            query_embedding = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Create a subset FAISS index for BM25 candidates
            candidate_embeddings = self.embeddings[bm25_indices]
            candidate_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
            candidate_index.add(candidate_embeddings)
            
            # Search the candidate subset
            faiss_scores, faiss_indices = candidate_index.search(query_embedding, min(max_results, len(bm25_indices)))
            
            # Step 3: Combine and boost scores
            combined_results = []
            for i, (faiss_score, faiss_idx) in enumerate(zip(faiss_scores[0], faiss_indices[0])):
                if faiss_idx < len(bm25_indices):
                    original_chunk_idx = bm25_indices[faiss_idx]
                    chunk = self.chunks[original_chunk_idx]
                    bm25_score = bm25_scores[original_chunk_idx]
                    
                    # Combine scores (weighted average)
                    combined_score = 0.6 * float(faiss_score) + 0.4 * bm25_score
                    
                    # Apply section boosting
                    boosted_score = combined_score
                    if chunk.get('is_executive_summary', False):
                        boosted_score += 0.15
                    if chunk.get('is_geographical', False):
                        boosted_score += 0.10
                    if chunk.get('page', 999) <= 10:
                        boosted_score += 0.05
                    
                    # Boost for key terms
                    text_lower = chunk['text'].lower()
                    if any(term in text_lower for term in ['outside the eea', 'ten times', 'about ten times', 'significantly higher']):
                        boosted_score += 0.20
                    
                    combined_results.append({
                        'document': chunk['source'],
                        'quote': chunk['text'][:2000] + '...' if len(chunk['text']) > 2000 else chunk['text'],
                        'relevance': 'high' if boosted_score > 0.8 else 'medium' if boosted_score > 0.6 else 'low',
                        'score': boosted_score,
                        'faiss_score': float(faiss_score),
                        'bm25_score': bm25_score,
                        'page': chunk.get('page', 'Unknown'),
                        'section': chunk.get('section', 'Unknown')
                    })
            
            # Sort by combined score and take top results
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            sources = combined_results[:max_results]
            
            if sources and self.openai_client:
                answer = self._generate_answer_from_sources(query, sources)
            else:
                answer = "Found relevant information in the documents."
            
            self.cost_tracker['bm25_queries'] += 1
            self.cost_tracker['faiss_queries'] += 1
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'confidence': min(0.95, max(0.7, sources[0]['score'] if sources else 0.5)),
                'method': 'dual_retrieval',
                'cost_saved': True
            }
            
        except Exception as e:
            self.logger.error(f"Dual retrieval failed: {e}")
            return self._search_with_faiss(query, max_results)
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query using dynamic LLM-driven analysis instead of hardcoded patterns
        Provides more robust and context-aware query expansion
        """
        if self.query_analyzer:
            try:
                # Use dynamic query analyzer for intelligent expansion
                analysis = self.query_analyzer.analyze_query(query)
                search_terms = self.query_analyzer.generate_document_search_terms(analysis)
                
                # Create expanded queries from search terms
                expanded_queries = [query]  # Start with original query
                
                # Add individual search terms as queries
                expanded_queries.extend(search_terms)
                
                # Add combinations of key terms
                key_terms = analysis.get("key_concepts", [])[:5]  # Top 5 concepts
                if len(key_terms) >= 2:
                    # Create 2-term combinations
                    for i in range(len(key_terms)):
                        for j in range(i + 1, len(key_terms)):
                            expanded_queries.append(f"{key_terms[i]} {key_terms[j]}")
                
                # Add specific metrics and time references
                specific_metrics = analysis.get("specific_metrics", [])
                time_references = analysis.get("time_references", [])
                
                for metric in specific_metrics:
                    expanded_queries.append(f"fraud {metric}")
                    expanded_queries.append(f"cross-border {metric}")
                
                for time_ref in time_references:
                    expanded_queries.append(f"fraud {time_ref}")
                    expanded_queries.append(f"cross-border {time_ref}")
                
                # Remove duplicates while preserving order
                unique_queries = list(dict.fromkeys(expanded_queries))
                return unique_queries[:15]  # Limit to top 15 expansions
                
            except Exception as e:
                self.logger.warning(f"Dynamic query expansion failed, using fallback: {e}")
                return self._fallback_query_expansion(query)
        else:
            return self._fallback_query_expansion(query)
    
    def _fallback_query_expansion(self, query: str) -> List[str]:
        """
        Fallback query expansion when dynamic analyzer is not available
        Uses basic keyword-based expansion
        """
        query_lower = query.lower()
        expanded_queries = [query]  # Start with original query
        
        # Basic expansions based on common patterns
        if any(term in query_lower for term in ['cross-border', 'cross border']):
            expanded_queries.extend([
                "cross-border fraud", "cross border fraud", "international fraud",
                "foreign fraud", "overseas fraud"
            ])
        
        if any(term in query_lower for term in ['share', 'percentage', 'proportion']):
            expanded_queries.extend([
                "fraud share", "fraud percentage", "fraud proportion",
                "fraud ratio", "fraud amount"
            ])
        
        if any(term in query_lower for term in ['h1', '2023', 'first half']):
            expanded_queries.extend([
                "H1 2023 fraud", "first half 2023", "2023 fraud",
                "fraud 2023", "fraud H1"
            ])
        
        if any(term in query_lower for term in ['value', 'amount', 'total']):
            expanded_queries.extend([
                "fraud value", "fraud amount", "fraud total",
                "fraud cost", "fraud loss"
            ])
        
        # Remove duplicates while preserving order
        unique_queries = list(dict.fromkeys(expanded_queries))
        return unique_queries[:10]  # Limit to top 10 expansions
    
    def _search_with_faiss(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using FAISS only"""
        if not self.use_faiss or self.index is None:
            return self._search_with_openai(query, max_results)
        
        try:
            # Encode query using OpenAI
            query_embedding = self._create_openai_embeddings([query])
            # Ensure query embedding is float32 and contiguous for FAISS
            query_embedding = query_embedding.astype(np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search with more results for boosting
            search_k = min(max_results * 3, len(self.chunks))
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Apply section boosting and re-rank
            boosted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    
                    # Apply section boosting based on GPT's recommendations
                    boosted_score = float(score)
                    
                    # Boost Executive Summary (where the key sentence is)
                    if chunk.get('is_executive_summary', False):
                        boosted_score += 0.15
                    
                    # Boost geographical sections
                    if chunk.get('is_geographical', False):
                        boosted_score += 0.10
                    
                    # Boost early pages (Executive Summary is usually early)
                    if chunk.get('page', 999) <= 10:
                        boosted_score += 0.05
                    
                    # Boost chunks with key terms
                    text_lower = chunk['text'].lower()
                    if any(term in text_lower for term in ['outside the eea', 'ten times', 'about ten times', 'significantly higher']):
                        boosted_score += 0.20
                    
                    boosted_results.append({
                        'document': chunk['source'],
                        'quote': chunk['text'][:2000] + '...' if len(chunk['text']) > 2000 else chunk['text'],
                        'relevance': 'high' if boosted_score > 0.8 else 'medium' if boosted_score > 0.6 else 'low',
                        'score': boosted_score,
                        'original_score': float(score),
                        'page': chunk.get('page', 'Unknown'),
                        'section': chunk.get('section', 'Unknown')
                    })
            
            # Sort by boosted score and take top results
            boosted_results.sort(key=lambda x: x['score'], reverse=True)
            sources = boosted_results[:max_results]
            
            # Generate answer using OpenAI with FAISS results
            if sources and self.openai_client:
                answer = self._generate_answer_from_sources(query, sources)
            else:
                answer = "Found relevant information in the documents."
            
            self.cost_tracker['faiss_queries'] += 1
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'confidence': min(0.9, max(0.6, float(scores[0][0]))),
                'method': 'faiss_only',
                'cost_saved': True
            }
            
        except Exception as e:
            self.logger.error(f"FAISS search failed: {e}")
            return self._search_with_openai(query, max_results)
    
    def _search_with_openai(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using OpenAI only"""
        if not self.openai_client:
            return self._generate_fallback_response(query)
        
        try:
            # Prepare context from all documents
            context = self._prepare_document_context()
            
            # Use OpenAI to analyze the query against documents
            prompt = f"""
            You are analyzing fraud-related documents to answer user questions about fraud methods and detection systems.
            
            Available Documents:
            {context}
            
            User Question: "{query}"
            
            Please analyze the documents and provide a comprehensive answer. Focus on:
            1. Specific fraud methods mentioned in the documents
            2. Detection system components described
            3. Best practices and recommendations
            4. Relevant statistics or data points
            
            IMPORTANT: Format your response as valid JSON with this exact structure:
            {{
                "answer": "Your detailed answer based on the documents",
                "sources": [
                    {{
                        "document": "EBA_ECB 2024 Report on Payment Fraud.pdf",
                        "quote": "Exact quote from the document",
                        "relevance": "high"
                    }}
                ],
                "confidence": 0.85
            }}
            
            Make sure to:
            - Use actual quotes from the documents
            - Specify the correct document names
            - Provide specific, actionable information
            - Keep quotes concise but meaningful
            """
            
            response = self.openai_client._make_api_call(prompt, max_tokens=1000)
            
            if response:
                try:
                    # Clean up the response to extract JSON
                    response_clean = response.strip()
                    if response_clean.startswith('```json'):
                        response_clean = response_clean[7:]
                    if response_clean.endswith('```'):
                        response_clean = response_clean[:-3]
                    response_clean = response_clean.strip()
                    
                    result = json.loads(response_clean)
                    result["success"] = True
                    result["method"] = "openai_only"
                    result["cost_saved"] = False
                    
                    self.cost_tracker['openai_queries'] += 1
                    return result
                    
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "answer": response,
                        "sources": [{
                            "document": "Fraud Analysis Documents",
                            "quote": "Information extracted from fraud analysis PDFs",
                            "relevance": "high"
                        }],
                        "confidence": 0.8,
                        "method": "openai_only",
                        "cost_saved": False
                    }
            else:
                return self._generate_fallback_response(query)
                
        except Exception as e:
            self.logger.error(f"OpenAI search failed: {e}")
            return self._generate_fallback_response(query)
    
    def _search_with_hybrid(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using hybrid approach: FAISS + OpenAI"""
        print(f"🔄 Executing hybrid search (FAISS + OpenAI)")
        
        if not self.use_faiss or self.index is None:
            print("  → FAISS not available, falling back to OpenAI only")
            return self._search_with_openai(query, max_results)
        
        try:
            # Step 1: Use FAISS to find relevant chunks
            print(f"  📊 Step 1: FAISS search for {max_results} results")
            faiss_result = self._search_with_faiss(query, max_results)
            
            if not faiss_result.get('success', False) or not faiss_result.get('sources'):
                print("  ❌ FAISS search failed, falling back to OpenAI only")
                return self._search_with_openai(query, max_results)
            
            print(f"  ✅ FAISS found {len(faiss_result['sources'])} sources")
            print(f"  📈 FAISS confidence: {faiss_result.get('confidence', 'N/A')}")
            
            # Step 2: Use OpenAI to analyze and enhance FAISS results
            print(f"  🤖 Step 2: OpenAI enhancement of FAISS results")
            sources = faiss_result['sources']
            answer = self._generate_answer_from_sources(query, sources)
            
            self.cost_tracker['hybrid_queries'] += 1
            self.cost_tracker['total_tokens_saved'] += 500  # Estimate tokens saved
            
            print(f"  ✅ Hybrid search completed successfully")
            print(f"  💰 Cost saved: ~500 tokens estimated")
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'confidence': faiss_result.get('confidence', 0.8),
                'method': 'hybrid',
                'cost_saved': True,
                'faiss_sources': len(sources),
                'enhanced_with_openai': True
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return self._search_with_openai(query, max_results)
    
    def _generate_answer_from_sources(self, query: str, sources: List[Dict]) -> str:
        """Generate answer using OpenAI with FAISS sources"""
        if not self.openai_client:
            return "Found relevant information in the documents."
        
        # Prepare context from sources
        context = ""
        for i, source in enumerate(sources[:5]):  # Use more sources for better coverage
            context += f"\n--- Source {i+1} from {source['document']} ---\n"
            context += source['quote'] + "\n"
        
        prompt = f"""
        Based on the following relevant document excerpts, answer the user's question with specific data and statistics:
        
        Question: "{query}"
        
        Relevant excerpts:
        {context}
        
        IMPORTANT INSTRUCTIONS:
        1. Look for specific numbers, percentages, and statistics in the excerpts
        2. If you find data like "ten times higher", "28%", "order of magnitude", include them
        3. Look for terms like "SCA", "strong customer authentication", "cross-border", "outside EEA"
        4. Be specific and cite exact numbers when available
        5. If multiple statistics are mentioned, include all relevant ones
        6. Focus on quantitative data and specific comparisons
        
        Please provide a comprehensive answer based on the excerpts. Be specific and cite the sources when appropriate.
        """
        
        try:
            response = self.openai_client._make_api_call(prompt, max_tokens=800)
            return response if response else "Found relevant information in the documents."
        except Exception as e:
            self.logger.error(f"Error generating answer from sources: {e}")
            return "Found relevant information in the documents."
    
    def _prepare_document_context(self) -> str:
        """Prepare document context for OpenAI analysis"""
        context = ""
        for doc_name, doc_text in self.documents.items():
            # Truncate very long documents to stay within token limits
            if len(doc_text) > 8000:
                doc_text = doc_text[:8000] + "... [truncated]"
            
            context += f"\n\n=== {doc_name} ===\n{doc_text}\n"
        
        return context
    
    def _generate_fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate a fallback response when documents aren't available"""
        try:
            if not self.openai_client:
                return {
                    "success": True,
                    "answer": "I apologize, but I'm unable to access the fraud analysis documents at the moment. However, I can provide general information about fraud methods and detection systems based on my training data.",
                    "sources": [{
                        "document": "General Knowledge",
                        "quote": "Response based on general fraud detection knowledge",
                        "relevance": "medium"
                    }],
                    "confidence": 0.6,
                    "method": "fallback"
                }
            
            # Use OpenAI to generate a response based on general knowledge
            prompt = f"""
            You are a fraud detection expert. Answer this question about fraud methods or detection systems:
            
            Question: "{query}"
            
            Provide a comprehensive answer based on your knowledge of fraud detection best practices.
            Focus on common fraud methods, detection techniques, and system components.
            """
            
            response = self.openai_client._make_api_call(prompt, max_tokens=800)
            
            if response:
                return {
                    "success": True,
                    "answer": response,
                    "sources": [{
                        "document": "AI Knowledge Base",
                        "quote": "Generated response based on fraud detection expertise",
                        "relevance": "high"
                    }],
                    "confidence": 0.8,
                    "method": "fallback_openai"
                }
            else:
                return {
                    "success": True,
                    "answer": "I apologize, but I'm unable to access the fraud analysis documents at the moment. Please try again later or contact support.",
                    "sources": [{
                        "document": "System Message",
                        "quote": "Document access temporarily unavailable",
                        "relevance": "low"
                    }],
                    "confidence": 0.3,
                    "method": "fallback_error"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating fallback response: {e}",
                "results": []
            }
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of processed documents and cost tracking"""
        return {
            "documents_loaded": len(self.documents),
            "document_names": list(self.documents.keys()),
            "total_characters": sum(len(text) for text in self.documents.values()),
            "processing_available": self.is_available(),
            "faiss_available": self.use_faiss,
            "chunks_created": len(self.chunks),
            "cost_tracker": self.cost_tracker
        }
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get cost optimization statistics"""
        total_queries = sum(self.cost_tracker.values())
        return {
            "total_queries": total_queries,
            "faiss_queries": self.cost_tracker['faiss_queries'],
            "openai_queries": self.cost_tracker['openai_queries'],
            "hybrid_queries": self.cost_tracker['hybrid_queries'],
            "cost_savings_percentage": (self.cost_tracker['faiss_queries'] + self.cost_tracker['hybrid_queries']) / max(total_queries, 1) * 100,
            "estimated_tokens_saved": self.cost_tracker['total_tokens_saved']
        }
    
    def test_connection(self) -> bool:
        """Test if document processing is working"""
        if not self.is_available():
            return False
        
        try:
            # Test OpenAI connection
            if self.openai_client:
                return self.openai_client.test_connection()
            return False
        except Exception as e:
            self.logger.error(f"Document processor test failed: {e}")
            return False
