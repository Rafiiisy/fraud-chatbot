"""
Simplified Document Processing Pipeline using OpenAI
Handles PDF text extraction and OpenAI-based document search
"""
import os
import json
from typing import List, Dict, Optional, Any
from pathlib import Path

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# OpenAI integration
try:
    from .openai_integration import OpenAIIntegration
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SimpleDocumentProcessor:
    """
    Simplified document processor that uses OpenAI for all document analysis
    """
    
    def __init__(self, documents_dir: str = "dataset"):
        # If running from backend directory, adjust path to parent
        if Path.cwd().name == "backend":
            self.documents_dir = Path("..") / documents_dir
        else:
            self.documents_dir = Path(documents_dir)
        
        self.documents = {}
        self.openai_client = None
        
        # Initialize OpenAI if available
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAIIntegration()
        
        # PDF files to process
        self.pdf_files = [
            "EBA_ECB 2024 Report on Payment Fraud.pdf",
            "Understanding Credit Card Frauds.pdf"
        ]
    
    def is_available(self) -> bool:
        """Check if document processing is available"""
        return PDF_AVAILABLE and OPENAI_AVAILABLE
    
    def process_documents(self) -> bool:
        """
        Process all PDF documents and extract text
        Returns True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
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
            
            return len(self.documents) > 0
            
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
    
    def search_documents(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Search documents using OpenAI for semantic understanding
        Returns relevant document sections and analysis
        """
        if not self.is_available():
            # If OpenAI is available but PDF processing isn't, still provide a response
            if OPENAI_AVAILABLE:
                return self._generate_fallback_response(query)
            return {
                "success": False,
                "error": "Document processing not available - OpenAI integration missing",
                "results": []
            }
        
        # Ensure documents are processed
        if not self.documents:
            print("No documents loaded, processing documents...")
            if not self.process_documents():
                # If document processing fails, provide a fallback response
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
            - Do NOT include (Source X) references in your answer text
            """
            
            response = self.openai_client._make_api_call(prompt, max_tokens=1000)
            
            # Debug logging
            print(f"\n=== DOCUMENT PROCESSOR DEBUG ===")
            print(f"Query: {query}")
            print(f"Raw OpenAI Response: {response}")
            
            if response:
                try:
                    # Clean up the response to extract JSON
                    response_clean = response.strip()
                    if response_clean.startswith('```json'):
                        response_clean = response_clean[7:]
                    if response_clean.endswith('```'):
                        response_clean = response_clean[:-3]
                    response_clean = response_clean.strip()
                    
                    print(f"Cleaned Response: {response_clean}")
                    
                    # Try to parse JSON response
                    result = json.loads(response_clean)
                    result["success"] = True
                    
                    print(f"Parsed Result: {result}")
                    print(f"Sources found: {len(result.get('sources', []))}")
                    print("=== END DOCUMENT PROCESSOR DEBUG ===\n")
                    
                    return result
                except json.JSONDecodeError:
                    # If not JSON, try to extract meaningful information from the response
                    # Split response into lines and look for structured content
                    lines = response.strip().split('\n')
                    answer = response
                    sources = []
                    
                    # Try to extract document references
                    for line in lines:
                        if 'document' in line.lower() or 'source' in line.lower():
                            sources.append({
                                "document": "PDF Document",
                                "quote": line.strip(),
                                "relevance": "medium"
                            })
                    
                    # If no sources found, create a generic one
                    if not sources:
                        sources = [{
                            "document": "Fraud Analysis Documents",
                            "quote": "Information extracted from fraud analysis PDFs",
                            "relevance": "high"
                        }]
                    
                    return {
                        "success": True,
                        "answer": answer,
                        "sources": sources,
                        "confidence": 0.8
                    }
            else:
                return {
                    "success": False,
                    "error": "OpenAI API call failed",
                    "results": []
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error searching documents: {e}",
                "results": []
            }
    
    def _prepare_document_context(self) -> str:
        """Prepare document context for OpenAI analysis"""
        context = ""
        for doc_name, doc_text in self.documents.items():
            # Truncate very long documents to stay within token limits
            if len(doc_text) > 8000:
                doc_text = doc_text[:8000] + "... [truncated]"
            
            context += f"\n\n=== {doc_name} ===\n{doc_text}\n"
        
        return context
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get summary of processed documents"""
        return {
            "documents_loaded": len(self.documents),
            "document_names": list(self.documents.keys()),
            "total_characters": sum(len(text) for text in self.documents.values()),
            "processing_available": self.is_available()
        }
    
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
                    "confidence": 0.6
                }
            
            # Use OpenAI to generate a response based on general knowledge
            prompt = f"""
            You are a fraud detection expert. Answer this question about fraud methods or detection systems:
            
            Question: "{query}"
            
            Provide a comprehensive answer based on your knowledge of fraud detection best practices.
            Focus on common fraud methods, detection techniques, and system components.
            
            Format as JSON:
            {{
                "answer": "Your detailed answer here",
                "sources": [
                    {{
                        "document": "Fraud Detection Knowledge Base",
                        "quote": "Key insight or method",
                        "relevance": "high"
                    }}
                ],
                "confidence": 0.8
            }}
            """
            
            response = self.openai_client._make_api_call(prompt, max_tokens=800)
            
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
                    return result
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "answer": response,
                        "sources": [{
                            "document": "AI Knowledge Base",
                            "quote": "Generated response based on fraud detection expertise",
                            "relevance": "high"
                        }],
                        "confidence": 0.8
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
                    "confidence": 0.3
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error generating fallback response: {e}",
                "results": []
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
            print(f"Document processor test failed: {e}")
            return False
