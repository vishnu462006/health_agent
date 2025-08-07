# Health Insurance Query Assistant

## Overview

This is a document-based question-answering system designed specifically for health insurance queries. The application allows users to upload insurance documents (PDF, DOCX, MSG, EML formats) and ask natural language questions about their content. It uses semantic search with vector embeddings to find relevant document sections and provides AI-generated answers based on the retrieved context.

The system combines document processing, vector search capabilities, and large language model integration to create an intelligent assistant that can understand and respond to complex insurance-related questions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Web Interface**: Single-page application built with HTML, CSS, and JavaScript
- **UI Framework**: Bootstrap 5 for responsive design and components
- **File Upload**: Drag-and-drop interface with progress tracking
- **Real-time Updates**: Dynamic content updates without page refreshes

### Backend Architecture
- **Web Framework**: FastAPI for high-performance API development
- **Document Processing Pipeline**: Multi-format document parser supporting PDF, DOCX, MSG, and EML files
- **Text Processing**: NLTK for sentence tokenization and language detection
- **Chunking Strategy**: Intelligent text segmentation for optimal retrieval performance

### Vector Search System
- **Embedding Model**: SentenceTransformers (all-MiniLM-L6-v2) for semantic text representation
- **Vector Database**: Pinecone for scalable similarity search
- **Index Configuration**: 384-dimensional vectors with cosine similarity metric
- **Retrieval Strategy**: Semantic search with configurable result limits

### AI Integration
- **LLM Provider**: OpenRouter API for accessing various language models
- **Default Model**: GPT-4o-mini for cost-effective performance
- **Context-Aware Responses**: RAG (Retrieval-Augmented Generation) pattern for accurate answers
- **Fallback Handling**: Graceful degradation when external services are unavailable

### Configuration Management
- **Environment Variables**: Secure API key and configuration management
- **Default Values**: Fallback configurations for development and testing
- **Service Endpoints**: Configurable external service URLs and parameters

## External Dependencies

### Vector Database
- **Pinecone**: Cloud-based vector database for semantic search
- **Configuration**: AWS us-east-1 region with serverless deployment
- **Index Management**: Automatic index creation and management

### AI Services
- **OpenRouter**: API gateway for accessing multiple LLM providers
- **Models**: Support for various models with GPT-4o-mini as default
- **Rate Limiting**: Built-in handling for API rate limits and quotas

### Document Processing Libraries
- **pdfplumber**: PDF text extraction and processing
- **python-docx**: Microsoft Word document parsing
- **extract-msg**: Outlook MSG file processing
- **email**: Standard library for EML email parsing

### Machine Learning
- **SentenceTransformers**: Pre-trained models for text embeddings
- **NLTK**: Natural language processing toolkit for text preprocessing
- **langdetect**: Language detection for multilingual support

### Web Framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for serving the FastAPI application
- **Jinja2**: Template engine for server-side rendering
- **Static Files**: CSS and JavaScript asset serving

### Development Tools
- **Requests**: HTTP client for external API communication
- **Pydantic**: Data validation and settings management
- **Tempfile**: Secure temporary file handling for uploads