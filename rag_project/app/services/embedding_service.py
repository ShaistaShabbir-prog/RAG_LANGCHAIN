from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os


class EmbeddingService:
    def __init__(self, embedding_model_name):
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_db = None

    def ingest_documents(self, file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        self.vector_db = FAISS.from_documents(texts, self.embedding_model)

    def retrieve(self, query, k=3):
        return self.vector_db.similarity_search(query, k=k) if self.vector_db else []


embedding_service = EmbeddingService(EMBEDDING_MODEL)

# embedding_service = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2")
