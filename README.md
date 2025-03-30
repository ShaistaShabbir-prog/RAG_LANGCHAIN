# RAG-Based Document Retrieval and Question Answering System

## Overview
This project implements a simplified Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant documents based on natural language queries and generates context-aware answers using a Large Language Model (LLM).

## Features
- **Document Embedding and Retrieval**
  - Uses a pre-trained embedding model to generate vector representations of text documents.
  - Stores and manages embeddings using a lightweight vector database.
  - Implements similarity search to retrieve the most relevant documents based on a user query.

- **Answer Generation Using LLM**
  - Uses an LLM to generate a response.
  - Provides the retrieved documents as context to improve the answer quality.

- **API Development**
  - Implements a FastAPI-based API with two key endpoints:
    - `/api/ingest`: Accepts documents and stores their embeddings.
    - `/api/query`: Accepts user queries, retrieves relevant documents, and generates an answer.

- **Containerization**
  - Dockerized application for easy deployment with a `Dockerfile` and `docker-compose.yml`.

## Installation

### Prerequisites
- Python 3.8+
- pip
- Docker (optional for containerized deployment)

### Setup
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd RAG_LANGCHAIN
   ```
2. Create a virtual environment:
   ```sh
   python -m venv rag_env
   source rag_env/bin/activate   # On Windows use: rag_env\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file:
   ```env
   HF_API_TOKEN=<your_huggingface_api_token>
   HF_MODEL=google/flan-t5-small
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   HF_API_URL=https://api-inference.huggingface.co/models/t5-small
   ```

5. Run the FastAPI server:
   ```sh
   uvicorn app.main:app --host 0.0.0.0 --port 10500 --reload
   ```

6. Access the API docs at:
   ```
   http://127.0.0.1:10500/docs
   ```

## Dockerization

### Build and Run with Docker
1. Build the Docker image:
   ```sh
   docker build -t rag_app .
   ```
2. Run the container:
   ```sh
   docker run -p 10500:10500 --env-file .env rag_app
   ```

### Using Docker Compose
1. Create a `docker-compose.yml` file:
   ```yaml
   version: '3.8'
   services:
     rag_service:
       build: .
       ports:
         - "10500:10500"
       env_file:
         - .env
   ```

2. Start the service:
   ```sh
   docker-compose up --build
   ```

## API Endpoints
### 1. Ingest Documents
- **Endpoint:** `POST /api/ingest`
- **Description:** Accepts documents and stores their embeddings.
- **Request Body:**
  ```json
  {
    "documents": ["Text of document 1", "Text of document 2"]
  }
  ```
- **Response:**
  ```json
  {
    "message": "Documents ingested successfully."
  }
  ```

### 2. Query the System
- **Endpoint:** `POST /api/query`
- **Description:** Accepts user queries, retrieves relevant documents, and generates an answer.
- **Request Body:**
  ```json
  {
    "query": "What is the capital of France?"
  }
  ```
- **Response:**
  ```json
  {
    "answer": "Paris"
  }
  ```

## Contributing
Feel free to submit pull requests or open issues for bug fixes, feature requests, or improvements.

## License
This project is licensed under the MIT License.

