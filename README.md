# RAG-pipeline
 
# RAG Pipeline – End-to-End (PDF → Retrieval → Reranking → Answer)

This project implements a  Retrieval Augmented Generation (RAG) pipeline from scratch.


---

## High-Level Architecture

PDF  
→ Text Extraction  
→ Chunking  
→ Embeddings  
→ FAISS HNSW Index  
→ Query Embedding  
→ ANN Retrieval  
→ Cross-Encoder Reranking  
→ LLM Answer Generation  

---

## Tech Stack

- Python 3.10  
- FAISS (HNSW index)  
- Sentence Transformers 
- Cross Encoder  
- FastAPI  
- Docker  
- AWS S3 (artifact storage)  
- AWS Lambda / API Gateway (explored, analyzed)  
- Kubernetes (local experimentation)

All models used are **free and open-source**.

---



