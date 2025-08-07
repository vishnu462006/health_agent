#!/usr/bin/env python3
"""
Health Insurance Query Assistant
A document-based question-answering system for health insurance queries.
Supports PDF, DOCX, MSG, EML file uploads with AI-powered responses.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import email
from email.policy import default
import traceback

# Core libraries
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Document processing
import pdfplumber
from docx import Document as DocxDocument
import extract_msg

# Text processing
import nltk
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


# Vector search (optional)
try:
    from sentence_transformers import SentenceTransformer
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    print("sentence-transformers or pinecone not available - vector search disabled")
    VECTOR_SEARCH_AVAILABLE = False

# === CONFIGURATION ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-3.5-turbo"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_7RNs5X_MyuECusJWqmjF7jzfWvxFSSYhtQ4zN4XzXCiCyibVhNBJejiZo5HaNtYaaEEmMU")
PINECONE_INDEX_NAME = "query-retrival-system"
PINECONE_ENVIRONMENT = "us-east-1"

# === PYDANTIC MODELS ===
class QueryRequest(BaseModel):
    query: str

class ParseResponse(BaseModel):
    parsed_data: Dict[str, Any]

class SearchResult(BaseModel):
    score: float
    text: str
    filename: str

class SearchResponse(BaseModel):
    results: List[SearchResult]

class AnswerResponse(BaseModel):
    answers: Any  # Can be single answer dict or list of answers

# === GLOBAL VARIABLES ===
app = FastAPI(title="Health Insurance Query Assistant")
model = None
index = None
pc = None

# Document store for fallback search
app.state.document_store = {}

# === INITIALIZE SERVICES ===
def init_pinecone():
    """Initialize Pinecone vector database"""
    global pc, index
    if not VECTOR_SEARCH_AVAILABLE:
        return False
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists, create if not
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # all-MiniLM-L6-v2 dimension
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        return True
    except Exception as e:
        print(f"Failed to connect to Pinecone: {e}")
        return False

def init_sentence_transformer():
    """Initialize sentence transformer model"""
    global model
    if not VECTOR_SEARCH_AVAILABLE:
        return False
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Successfully loaded sentence transformer model")
        return True
    except Exception as e:
        print(f"Failed to load sentence transformer: {e}")
        return False

def init_nltk():
    """Initialize NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# Initialize services
print(f"Using OpenRouter API key: {OPENROUTER_API_KEY[:20]}...{OPENROUTER_API_KEY[-10:]}")
init_nltk()
pinecone_success = init_pinecone()
transformer_success = init_sentence_transformer()

# === DOCUMENT PROCESSING FUNCTIONS ===
def process_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def process_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error processing DOCX: {str(e)}")

def process_msg(file_path: str) -> str:
    """Extract text from MSG file"""
    try:
        msg = extract_msg.Message(file_path)
        
        # Extract basic information
        subject = msg.subject or "No Subject"
        sender = msg.sender or "Unknown Sender"
        body = msg.body or ""
        
        # Create formatted text
        text = f"Subject: {subject}\n"
        text += f"From: {sender}\n"
        if msg.date:
            text += f"Date: {msg.date}\n"
        text += f"\nContent:\n{body}"
        
        msg.close()
        return text.strip()
    except Exception as e:
        raise Exception(f"Error processing MSG: {str(e)}")

def process_eml(file_path: str) -> str:
    """Extract text from EML file"""
    try:
        with open(file_path, 'rb') as f:
            msg = email.message_from_bytes(f.read(), policy=default)
        
        # Extract headers
        subject = msg.get('Subject', 'No Subject')
        sender = msg.get('From', 'Unknown Sender')
        date = msg.get('Date', '')
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.iter_parts():
                if part.get_content_type() == 'text/plain':
                    body += part.get_content()
                    break
        else:
            if msg.get_content_type() == 'text/plain':
                body = msg.get_content()
        
        # Create formatted text
        text = f"Subject: {subject}\n"
        text += f"From: {sender}\n"
        if date:
            text += f"Date: {date}\n"
        text += f"\nContent:\n{body}"
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error processing EML: {str(e)}")

def chunk_text(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into chunks with overlap"""
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    except Exception:
        # Fallback to simple word-based chunking
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_chunk_size):
            chunk = " ".join(words[i:i + max_chunk_size])
            chunks.append(chunk)
        return chunks

# === SEARCH FUNCTIONS ===
def semantic_search(query: str, top_k: int = 5) -> List[Dict]:
    """Perform semantic search using vector embeddings"""
    if not (model and index):
        return []
    
    try:
        # Create embedding for query
        query_embedding = model.encode([query])[0].tolist()
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        search_results = []
        for match in results.matches:
            search_results.append({
                "score": float(match.score),
                "text": match.metadata.get("text", ""),
                "filename": match.metadata.get("filename", "Unknown")
            })
        
        return search_results
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []

def parse_query_with_llm(query: str) -> Dict[str, Any]:
    """Parse query using LLM to extract structured information"""
    prompt = f"""
    Parse this health insurance question and extract structured information:
    
    Question: {query}
    
    Return a JSON object with these fields:
    - "intent": what the user wants to know (coverage, claim, benefit, etc.)
    - "keywords": important terms from the question
    - "category": type of insurance question (medical, dental, vision, etc.)
    
    Return only valid JSON.
    """
    
    try:
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a health insurance query parser. Return only JSON."},
                {"role": "user", "content": prompt}
            ]
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result:
            content = result["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback parsing
                return {
                    "intent": "general_inquiry",
                    "keywords": query.split(),
                    "category": "general"
                }
        else:
            raise Exception("Invalid response from OpenRouter")
            
    except Exception as e:
        print(f"Query parsing failed: {e}")
        # Return fallback structure
        return {
            "intent": "general_inquiry", 
            "keywords": query.split(),
            "category": "general"
        }

# === TEMPLATES AND STATIC FILES (EMBEDDED) ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Insurance Query Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body { background-color: #f8f9fa; }
        .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-bottom: 2rem; }
        .upload-area { border: 2px dashed #007bff; border-radius: 10px; padding: 2rem; text-align: center; background-color: #f8f9ff; transition: all 0.3s ease; cursor: pointer; }
        .upload-area:hover { border-color: #0056b3; background-color: #e6f0ff; }
        .upload-area.dragover { border-color: #28a745; background-color: #e6f7e6; }
        .feature-icon { font-size: 3rem; color: #007bff; margin-bottom: 1rem; }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
        .btn-primary:hover { background: linear-gradient(135deg, #5a67d8 0%, #6b46a3 100%); }
        .progress { height: 20px; margin-top: 1rem; }
        .card { border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem; }
        .info-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .answer-section { padding: 1rem 0; }
        #results { margin-top: 2rem; }
        .loading-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center; z-index: 9999; }
        .spinner { width: 3rem; height: 3rem; }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="main-header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1 class="display-5 fw-bold mb-0">
                        <i class="fas fa-shield-alt me-3"></i>
                        Health Insurance Assistant
                    </h1>
                    <p class="lead mb-0">Upload your insurance documents and get instant answers to your coverage questions</p>
                </div>
                <div class="col-md-4 text-end">
                    <i class="fas fa-file-medical feature-icon"></i>
                </div>
            </div>
        </div>
    </div>

    <div class="container">
        <!-- Upload Section -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <i class="fas fa-cloud-upload-alt me-2"></i>
                            Upload Insurance Documents
                        </h5>
                    </div>
                    <div class="card-body">
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="upload-area" id="uploadArea">
                                <i class="fas fa-file-upload fa-3x text-primary mb-3"></i>
                                <h5>Drag and drop your files here</h5>
                                <p class="text-muted">or click to browse</p>
                                <p class="small text-muted">Supports PDF, DOCX, MSG, and EML files</p>
                                <input type="file" id="fileInput" name="file" class="d-none" accept=".pdf,.docx,.msg,.eml" multiple>
                            </div>
                            <div id="uploadProgress" class="d-none">
                                <div class="progress">
                                    <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" style="width: 0%"></div>
                                </div>
                                <p id="progressText" class="text-center mt-2">Uploading...</p>
                            </div>
                            <button type="submit" class="btn btn-primary btn-lg mt-3">
                                <i class="fas fa-upload me-2"></i>
                                Upload Documents
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <!-- Query Section -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <i class="fas fa-question-circle me-2"></i>
                            Ask Questions About Your Coverage
                        </h5>
                    </div>
                    <div class="card-body">
                        <form id="queryForm">
                            <div class="mb-3">
                                <textarea id="queryText" class="form-control" rows="3" placeholder="Ask about your insurance coverage, benefits, waiting periods, exclusions, etc."></textarea>
                            </div>
                            <div class="row">
                                <div class="col-md-4">
                                    <button type="button" id="parseBtn" class="btn btn-outline-info w-100">
                                        <i class="fas fa-search me-2"></i>
                                        Parse Query
                                    </button>
                                </div>
                                <div class="col-md-4">
                                    <button type="button" id="searchBtn" class="btn btn-outline-success w-100">
                                        <i class="fas fa-file-search me-2"></i>
                                        Search Documents
                                    </button>
                                </div>
                                <div class="col-md-4">
                                    <button type="button" id="answerBtn" class="btn btn-primary w-100">
                                        <i class="fas fa-brain me-2"></i>
                                        Get AI Answer
                                    </button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div id="results"></div>
    </div>

    <!-- Loading Modal -->
    <div class="modal fade" id="loadingModal" tabindex="-1" data-bs-backdrop="static">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-body text-center p-4">
                    <div class="spinner-border text-primary spinner mb-3"></div>
                    <h5>Processing your request...</h5>
                    <p class="text-muted mb-0">Please wait while we analyze your documents</p>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadForm = document.getElementById('uploadForm');
        const uploadProgress = document.getElementById('uploadProgress');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        
        function handleDragOver(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        }
        
        function handleDragLeave(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        }
        
        function handleDrop(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            fileInput.files = files;
        }

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            const files = fileInput.files;
            
            if (files.length === 0) {
                showAlert('Please select files to upload', 'warning');
                return;
            }
            
            // Show progress
            uploadProgress.classList.remove('d-none');
            progressBar.style.width = '10%';
            progressText.textContent = 'Uploading files...';
            
            // Upload files one by one
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const singleFormData = new FormData();
                singleFormData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: singleFormData
                    });
                    
                    if (!response.ok) throw new Error('Upload failed');
                    
                    const progress = ((i + 1) / files.length) * 90;
                    progressBar.style.width = progress + '%';
                    progressText.textContent = `Processing ${file.name}...`;
                    
                } catch (error) {
                    showAlert(`Error uploading ${file.name}: ${error.message}`, 'danger');
                    uploadProgress.classList.add('d-none');
                    return;
                }
            }
            
            progressBar.style.width = '100%';
            progressText.textContent = 'Upload complete!';
            
            setTimeout(() => {
                uploadProgress.classList.add('d-none');
                progressBar.style.width = '0%';
                showAlert(`Successfully uploaded ${files.length} file(s)`, 'success');
                fileInput.value = '';
            }, 1000);
        });

        // Query handling
        document.getElementById('parseBtn').addEventListener('click', async () => {
            const query = document.getElementById('queryText').value.trim();
            if (!query) {
                showAlert('Please enter a question', 'warning');
                return;
            }
            
            showLoading();
            try {
                const response = await fetch('/parse', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                displayParseResults(data.parsed_data);
            } catch (error) {
                showAlert('Error parsing query: ' + error.message, 'danger');
            }
            hideLoading();
        });

        document.getElementById('searchBtn').addEventListener('click', async () => {
            const query = document.getElementById('queryText').value.trim();
            if (!query) {
                showAlert('Please enter a question', 'warning');
                return;
            }
            
            showLoading();
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                displaySearchResults(data.results);
            } catch (error) {
                showAlert('Error searching documents: ' + error.message, 'danger');
            }
            hideLoading();
        });

        document.getElementById('answerBtn').addEventListener('click', async () => {
            const query = document.getElementById('queryText').value.trim();
            if (!query) {
                showAlert('Please enter a question', 'warning');
                return;
            }
            
            showLoading();
            try {
                const response = await fetch('/answer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                displayAnswerResults(data.answers);
            } catch (error) {
                showAlert('Error getting answer: ' + error.message, 'danger');
            }
            hideLoading();
        });

        function displayParseResults(parsed) {
            const resultsDiv = document.getElementById('results');
            const html = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-cogs me-2"></i>Query Analysis</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4">
                                <strong>Intent:</strong> ${parsed.intent || 'Unknown'}
                            </div>
                            <div class="col-md-4">
                                <strong>Category:</strong> ${parsed.category || 'General'}
                            </div>
                            <div class="col-md-4">
                                <strong>Keywords:</strong> ${(parsed.keywords || []).join(', ')}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            resultsDiv.innerHTML = html;
        }

        function displaySearchResults(results) {
            const resultsDiv = document.getElementById('results');
            let html = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0"><i class="fas fa-search me-2"></i>Search Results (${results.length} found)</h6>
                    </div>
                    <div class="card-body">
            `;

            if (results.length === 0) {
                html += '<p class="text-muted">No relevant documents found.</p>';
            } else {
                results.forEach((result, index) => {
                    html += `
                        <div class="border-bottom pb-3 mb-3">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h6 class="mb-1">Result ${index + 1}</h6>
                                <span class="badge bg-primary">${(result.score * 100).toFixed(1)}% match</span>
                            </div>
                            <p class="mb-2">${result.text}</p>
                            <small class="text-muted">Source: ${result.filename}</small>
                        </div>
                    `;
                });
            }

            html += `
                    </div>
                </div>
            `;

            resultsDiv.innerHTML = html;
        }

        function displayAnswerResults(answers) {
            const resultsDiv = document.getElementById('results');
            
            // Handle both single answer and array of answers
            const answerArray = Array.isArray(answers) ? answers : [answers];
            
            let html = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">
                            <i class="fas fa-lightbulb me-2"></i>
                            Insurance Information
                        </h6>
                    </div>
                    <div class="card-body">
            `;

            answerArray.forEach((answerData, index) => {
                const answer = answerData.answer;
                
                if (answerData.message) {
                    html += `
                        <div class="alert alert-warning">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            ${answerData.message}
                        </div>
                    `;
                    return;
                }

                html += `
                    <div class="answer-section ${index < answerArray.length - 1 ? 'border-bottom pb-3 mb-3' : ''}">
                        ${answerData.question ? `<h6 class="text-primary mb-3">Q: ${answerData.question}</h6>` : ''}
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="info-card">
                                    <h6 class="fw-bold text-success">
                                        <i class="fas fa-file-contract me-2"></i>
                                        Policy
                                    </h6>
                                    <p class="mb-0">${answer.policy || 'Not specified'}</p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="info-card">
                                    <h6 class="fw-bold text-info">
                                        <i class="fas fa-heart me-2"></i>
                                        Benefit
                                    </h6>
                                    <p class="mb-0">${answer.benefit || 'Not specified'}</p>
                                </div>
                            </div>
                        </div>

                        ${answer.details ? `
                            <div class="mt-3">
                                <h6 class="fw-bold text-secondary">
                                    <i class="fas fa-info-circle me-2"></i>
                                    Details
                                </h6>
                                ${answer.details.description ? `<p><strong>Description:</strong> ${answer.details.description}</p>` : ''}
                                ${answer.details.waiting_period ? `<p><strong>Waiting Period:</strong> ${answer.details.waiting_period}</p>` : ''}
                                ${answer.details.coverage_amount ? `<p><strong>Coverage Amount:</strong> ${answer.details.coverage_amount}</p>` : ''}
                                
                                ${answer.details.exclusions && answer.details.exclusions.length > 0 ? `
                                    <div class="mt-2">
                                        <strong>Exclusions:</strong>
                                        <ul class="list-unstyled ms-3">
                                            ${answer.details.exclusions.map(exclusion => `<li><i class="fas fa-times text-danger me-2"></i>${exclusion}</li>`).join('')}
                                        </ul>
                                    </div>
                                ` : ''}
                                
                                ${answer.details.notes && answer.details.notes.length > 0 ? `
                                    <div class="mt-2">
                                        <strong>Additional Notes:</strong>
                                        <ul class="list-unstyled ms-3">
                                            ${answer.details.notes.map(note => `<li><i class="fas fa-sticky-note text-warning me-2"></i>${note}</li>`).join('')}
                                        </ul>
                                    </div>
                                ` : ''}
                            </div>
                        ` : ''}

                        ${answerData.sources && answerData.sources.length > 0 ? `
                            <div class="mt-3">
                                <small class="text-muted">
                                    <i class="fas fa-file me-1"></i>
                                    Sources: ${answerData.sources.join(', ')}
                                </small>
                            </div>
                        ` : ''}
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;

            resultsDiv.innerHTML = html;
        }

        // Utility functions
        function showAlert(message, type = 'info') {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `
                <div class="alert alert-${type} alert-dismissible fade show">
                    ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
        }

        function showLoading() {
            loadingModal.show();
        }

        function hideLoading() {
            loadingModal.hide();
        }
    </script>
</body>
</html>
"""

# === API ROUTES ===
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main application page"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process document files"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.msg', '.eml'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process based on file type
            if file_extension == '.pdf':
                text = process_pdf(tmp_file_path)
            elif file_extension == '.docx':
                text = process_docx(tmp_file_path)
            elif file_extension == '.msg':
                text = process_msg(tmp_file_path)
            elif file_extension == '.eml':
                text = process_eml(tmp_file_path)
            
            # Detect language
            try:
                language = detect(text[:1000])
            except LangDetectError:
                language = "en"
            
            # Store in fallback document store
            app.state.document_store[file.filename] = text
            
            # Process for vector storage if available
            if model and index:
                try:
                    chunks = chunk_text(text)
                    embeddings = model.encode(chunks)
                    
                    vectors = []
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        vectors.append({
                            "id": f"{file.filename}_{i}",
                            "values": embedding.tolist(),
                            "metadata": {
                                "text": chunk,
                                "filename": file.filename,
                                "chunk_id": i,
                                "language": language
                            }
                        })
                    
                    # Upload to Pinecone
                    index.upsert(vectors=vectors)
                    
                except Exception as e:
                    print(f"Vector storage failed: {e}")
                    # Continue with fallback storage
            
            return {
                "message": f"Successfully processed {file.filename}",
                "filename": file.filename,
                "text_length": len(text),
                "language": language,
                "chunks": len(chunk_text(text)) if VECTOR_SEARCH_AVAILABLE else "N/A"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/parse", response_model=ParseResponse)
async def parse_query(req: QueryRequest):
    """Parse query to extract structured information"""
    try:
        parsed = parse_query_with_llm(req.query)
        # Ensure we return a dict, not a list
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        return ParseResponse(parsed_data=parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing query: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search(req: QueryRequest):
    """Perform semantic search on uploaded documents"""
    try:
        query_str = req.query if isinstance(req.query, str) else req.query[0]
        
        if index and model:
            # Use vector search if available
            results = semantic_search(query_str)
        else:
            # Fallback to enhanced text search
            results = []
            if hasattr(app.state, 'document_store'):
                print(f"Searching in fallback store with {len(app.state.document_store)} documents")
                print(f"Available documents: {list(app.state.document_store.keys())}")
                query_lower = query_str.lower()
                query_words = query_lower.split()
                
                for filename, text in app.state.document_store.items():
                    text_lower = text.lower()
                    relevance_score = 0
                    
                    # Calculate relevance based on word matches
                    for word in query_words:
                        if word in text_lower:
                            relevance_score += text_lower.count(word)
                    
                    if relevance_score > 0:
                        # Find the best matching section
                        best_section = ""
                        text_words = text.split()
                        for i, word in enumerate(text_words):
                            if any(qw in word.lower() for qw in query_words):
                                start = max(0, i - 30)
                                end = min(len(text_words), i + 100)
                                section = " ".join(text_words[start:end])
                                if len(section) > len(best_section):
                                    best_section = section
                                break
                        
                        if not best_section:
                            best_section = text[:500]
                        
                        results.append({
                            "score": min(relevance_score / len(text.split()) * 10, 1.0),
                            "text": best_section + ("..." if len(best_section) >= 500 else ""),
                            "filename": filename
                        })
                
                results = sorted(results, key=lambda x: x["score"], reverse=True)[:5]
        
        return SearchResponse(
            results=[
                SearchResult(score=r["score"], text=r["text"], filename=r["filename"]) 
                for r in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

@app.post("/answer", response_model=AnswerResponse)
async def get_answer(req: QueryRequest):
    """Get detailed answers using LLM based on search results"""
    try:
        queries = req.query if isinstance(req.query, list) else [req.query]
        all_answers = []

        for query in queries:
            # Try vector search first, then fallback to keyword search
            if index and model:
                results = semantic_search(query, top_k=3)
            else:
                # Fallback search when vector database is not available
                results = []
                if hasattr(app.state, 'document_store'):
                    query_lower = query.lower()
                    for filename, text in app.state.document_store.items():
                        # Enhanced keyword search with multiple terms
                        query_words = query_lower.split()
                        relevance_score = 0
                        text_lower = text.lower()
                        
                        # Calculate relevance based on word matches
                        for word in query_words:
                            if word in text_lower:
                                relevance_score += text_lower.count(word)
                        
                        if relevance_score > 0:
                            # Find the best matching section (first 1000 chars containing query words)
                            best_section = ""
                            text_words = text.split()
                            for i, word in enumerate(text_words):
                                if any(qw in word.lower() for qw in query_words):
                                    start = max(0, i - 50)
                                    end = min(len(text_words), i + 150)
                                    section = " ".join(text_words[start:end])
                                    if len(section) > len(best_section):
                                        best_section = section
                                    break
                            
                            if not best_section:
                                best_section = text[:800]
                            
                            results.append({
                                "score": min(relevance_score / len(text.split()) * 10, 1.0),
                                "text": best_section,
                                "filename": filename
                            })
                    
                    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
            
            if not results:
                all_answers.append({
                    "question": query,
                    "answer": {
                        "policy": None,
                        "benefit": None,
                        "details": {}
                    },
                    "message": "Sorry, no relevant information found in the uploaded documents."
                })
                continue

            # Combine top results for context
            context = "\n\n".join([f"From {r['filename']}: {r['text']}" for r in results[:3]])

            prompt = f"""
You are an expert health insurance assistant. Based on the policy content below, answer the user's question with a detailed JSON object in this format:

{{
  "policy": "Policy Name here",
  "benefit": "Benefit name here",
  "details": {{
    "waiting_period": "value if relevant",
    "description": "detailed description",
    "coverage_amount": "amount if mentioned",
    "exclusions": ["exclusion 1", "exclusion 2"],
    "notes": [
      "Note 1",
      "Note 2"
    ]
  }}
}}

Policy Content:
{context}

User's Question: {query}

Provide only the JSON object as the answer.
"""

            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You answer clearly with structured JSON as instructed."},
                    {"role": "user", "content": prompt}
                ]
            }
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            try:
                llm_response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
                llm_response.raise_for_status()
                response_json = llm_response.json()
                
                if "choices" not in response_json:
                    raise Exception(f"OpenRouter response missing 'choices': {response_json}")
                
                answer_text = response_json["choices"][0]["message"]["content"]
                
                try:
                    answer_json = json.loads(answer_text)
                except Exception:
                    answer_json = {
                        "policy": "Policy Information",
                        "benefit": "General Information",
                        "details": {
                            "description": answer_text,
                            "notes": ["Raw LLM response - could not parse as JSON"]
                        }
                    }
                
                all_answers.append({
                    "question": query,
                    "answer": answer_json,
                    "sources": [r["filename"] for r in results[:3]]
                })
                
            except Exception as e:
                print(f"LLM answer generation failed: {e}")
                all_answers.append({
                    "question": query,
                    "answer": {
                        "policy": "Error",
                        "benefit": "Could not generate answer",
                        "details": {
                            "description": f"Error generating answer: {str(e)}",
                            "notes": ["Please try rephrasing your question"]
                        }
                    },
                    "sources": [r["filename"] for r in results[:3]]
                })

        return AnswerResponse(answers=all_answers[0] if len(all_answers) == 1 else all_answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pinecone_connected": index is not None,
        "model_loaded": model is not None
    }

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    # For hosting platforms, bind to 0.0.0.0 and port 5000 for frontend access
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")