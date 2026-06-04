# AI Resume Search & Candidate Ranking System

## Features
- Resume upload (PDF)
- Semantic search using embeddings
- RAG question answering
- Job-to-candidate matching
- Multi-candidate ranking
- AWS S3 document storage
- Docker deployment

## Tech Stack
- Python
- Django REST Framework
- PostgreSQL
- pgvector
- OpenAI Embeddings
- Cohere Rerank
- AWS S3
- Docker

## Architecture
(upload architecture diagram)

## API Endpoints
POST /api/upload-resume/
POST /api/search-resume/
POST /api/ask/
POST /api/match-job/
POST /api/rank-candidates/

## Running Locally
docker compose up --build
