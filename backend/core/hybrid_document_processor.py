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

# OpenAI integration
try:
    from .openai_integration import OpenAIIntegration
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
        self.documents = {}
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.openai_client = None
        
        # Initialize components
        # Note: Using OpenAI embeddings instead of sentence-transformers
        
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAIIntegration()
        
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
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            return False
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
        except Exception as e:
            print(f"pdfplumber failed for {pdf_path}: {e}")
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text.strip()
            except Exception as e2:
                print(f"PyPDF2 also failed for {pdf_path}: {e2}")
                return ""
    
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
    
    def _chunk_document(self, text: str, doc_name: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Chunk document into smaller pieces for better search"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Only include substantial chunks
                chunks.append({
                    'text': chunk_text,
                    'source': doc_name,
                    'chunk_id': len(chunks),
                    'start_word': i,
                    'end_word': min(i + chunk_size, len(words))
                })
        
        return chunks
    
    def _create_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.openai_client.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                # Fallback: create random embeddings
                batch_embeddings = [np.random.rand(1536).tolist() for _ in batch]
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def search_documents(self, query: str, max_results: int = 5, use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Search documents using hybrid approach
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
            if search_strategy == 'faiss_only':
                return self._search_with_faiss(query, max_results)
            elif search_strategy == 'openai_only':
                return self._search_with_openai(query, max_results)
            else:  # hybrid
                return self._search_with_hybrid(query, max_results)
                
        except Exception as e:
            self.logger.error(f"Error in document search: {e}")
            return self._generate_fallback_response(query)
    
    def _determine_search_strategy(self, query: str, use_hybrid: bool) -> str:
        """
        Determine the best search strategy based on query characteristics
        """
        if not use_hybrid or not self.use_faiss:
            return 'openai_only'
        
        # Simple heuristics for routing
        query_lower = query.lower()
        
        # FAISS is good for:
        # - Specific fact-finding questions
        # - Short queries
        # - Questions asking for specific information
        faiss_indicators = [
            'what is', 'what are', 'define', 'explain', 'describe',
            'how does', 'how do', 'which', 'where', 'when',
            'list', 'name', 'identify', 'find'
        ]
        
        # OpenAI is better for:
        # - Complex analytical questions
        # - Questions requiring synthesis
        # - Long, detailed queries
        openai_indicators = [
            'analyze', 'compare', 'evaluate', 'assess', 'discuss',
            'why', 'what are the implications', 'what are the benefits',
            'what are the challenges', 'what are the risks'
        ]
        
        # Check query length
        if len(query.split()) < 5:
            return 'faiss_only'
        elif len(query.split()) > 15:
            return 'openai_only'
        
        # Check for specific indicators
        if any(indicator in query_lower for indicator in faiss_indicators):
            return 'faiss_only'
        elif any(indicator in query_lower for indicator in openai_indicators):
            return 'openai_only'
        
        # Default to hybrid for balanced approach
        return 'hybrid'
    
    def _search_with_faiss(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using FAISS only"""
        if not self.use_faiss or self.index is None:
            return self._search_with_openai(query, max_results)
        
        try:
            # Encode query using OpenAI
            query_embedding = self._create_openai_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, max_results)
            
            # Format results
            sources = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    sources.append({
                        'document': chunk['source'],
                        'quote': chunk['text'][:500] + '...' if len(chunk['text']) > 500 else chunk['text'],
                        'relevance': 'high' if score > 0.7 else 'medium' if score > 0.5 else 'low',
                        'score': float(score)
                    })
            
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
        if not self.use_faiss or self.index is None:
            return self._search_with_openai(query, max_results)
        
        try:
            # Step 1: Use FAISS to find relevant chunks
            faiss_result = self._search_with_faiss(query, max_results)
            
            if not faiss_result.get('success', False) or not faiss_result.get('sources'):
                return self._search_with_openai(query, max_results)
            
            # Step 2: Use OpenAI to analyze and enhance FAISS results
            sources = faiss_result['sources']
            answer = self._generate_answer_from_sources(query, sources)
            
            self.cost_tracker['hybrid_queries'] += 1
            self.cost_tracker['total_tokens_saved'] += 500  # Estimate tokens saved
            
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
        for i, source in enumerate(sources[:3]):  # Limit to top 3 sources
            context += f"\n--- Source {i+1} from {source['document']} ---\n"
            context += source['quote'] + "\n"
        
        prompt = f"""
        Based on the following relevant document excerpts, answer the user's question:
        
        Question: "{query}"
        
        Relevant excerpts:
        {context}
        
        Please provide a comprehensive answer based on the excerpts. Be specific and cite the sources when appropriate.
        """
        
        try:
            response = self.openai_client._make_api_call(prompt, max_tokens=600)
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
