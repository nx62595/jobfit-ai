# AI Resume Search & Candidate Ranking System

## Highlights

- Built a Retrieval-Augmented Generation (RAG) recruiting platform using Django, PostgreSQL, pgvector, OpenAI Embeddings, and Cohere Rerank.
- Implemented semantic resume search, candidate Q&A, job matching, and multi-candidate ranking.
- Designed a deterministic scoring system to improve ranking consistency and reduce LLM scoring variability.
- Dockerized the full application stack and integrated AWS S3 for document storage.

## Features
- Upload PDF resumes
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

![Architecture](docs/images/architecture.png)

## API Endpoints

```text
POST /api/upload-resume/
POST /api/search-resume/
POST /api/ask/
POST /api/match-job/
POST /api/rank-candidates/
```

## Running Locally

1. Clone the repository

```bash
git clone https://github.com/nx62595/jobfit-ai.git
cd jobfit-ai
```

2. Create environment variables

```bash
cp backend/.env.example backend/.env
```

3. Start services

```bash
docker compose up --build
```

4. API available at

```text
http://localhost:8000
```

## Resume Upload

![Resume Upload](docs/images/resume-upload-api.png)

## Candidate Ranking

![Candidate Ranking](docs/images/candidate-ranking-results.png)

## Candidate Q&A

![Candidate Q&A](docs/images/candidate-question-answering.png)
