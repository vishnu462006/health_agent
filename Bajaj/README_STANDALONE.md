# Health Insurance Query Assistant - Standalone Version

A complete health insurance document query assistant that allows users to upload insurance documents (PDF, DOCX, MSG, EML) and ask AI-powered questions about their coverage, benefits, and policy details.

## Features

✅ **Document Upload System**: Supports PDF, DOCX, MSG, and EML files  
✅ **AI-Powered Q&A**: Uses OpenRouter API for intelligent responses  
✅ **Vector Search**: Pinecone integration with fallback keyword search  
✅ **Clean Web Interface**: Bootstrap-powered responsive design  
✅ **Structured Responses**: JSON-formatted insurance information  
✅ **Fallback Systems**: Works even without vector database  

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn requests pdfplumber python-docx extract-msg nltk langdetect pydantic python-multipart
```

### Optional (for vector search):
```bash
pip install sentence-transformers pinecone
```

### 2. Set Environment Variables

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"  # optional
```

### 3. Run the Application

```bash
python health_insurance_assistant.py
```

The app will start on http://localhost:5000

## Usage

1. **Upload Documents**: Drag and drop your insurance documents (PDF, DOCX, MSG, EML)
2. **Ask Questions**: Type questions about your coverage, benefits, waiting periods, etc.
3. **Get AI Answers**: Receive structured responses with policy details, benefits, and exclusions

## Example Questions

- "What is the waiting period for dental coverage?"
- "What does my policy cover for emergency room visits?"
- "Are there any exclusions for pre-existing conditions?"
- "What is my maximum annual benefit for vision care?"

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload insurance documents
- `POST /parse` - Parse query structure
- `POST /search` - Search documents
- `POST /answer` - Get AI-powered answers
- `GET /health` - Health check

## File Structure

The single file `health_insurance_assistant.py` contains:

- FastAPI web server
- Document processing (PDF, DOCX, MSG, EML)
- Vector search with Pinecone (optional)
- Fallback keyword search
- OpenRouter LLM integration
- Complete HTML/CSS/JavaScript frontend
- All API endpoints

## Configuration

### Required:
- `OPENROUTER_API_KEY`: Your OpenRouter API key for AI responses

### Optional:
- `PINECONE_API_KEY`: Pinecone API key for vector search
- `PINECONE_INDEX_NAME`: Index name (default: "query-retrival-system")
- `PINECONE_ENVIRONMENT`: Environment (default: "us-east-1")

## Dependencies

### Core (Required):
- fastapi - Web framework
- uvicorn - ASGI server
- requests - HTTP client
- pdfplumber - PDF processing
- python-docx - Word document processing
- extract-msg - Outlook MSG files
- nltk - Text processing
- langdetect - Language detection
- pydantic - Data validation
- python-multipart - File uploads

### Optional (Enhanced Features):
- sentence-transformers - Text embeddings
- pinecone - Vector database

## Notes

- The application works with or without vector search capabilities
- If Pinecone/sentence-transformers are not available, it uses enhanced keyword search
- All frontend assets (HTML, CSS, JavaScript) are embedded in the Python file
- Supports multiple file formats commonly used for insurance documents
- Provides structured JSON responses for easy integration

## Error Handling

The application includes comprehensive error handling:
- File upload validation
- Document processing fallbacks
- API timeout handling
- Graceful degradation when services are unavailable

## License

Open source - feel free to modify and use as needed.