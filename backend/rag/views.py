from pypdf import PdfReader
from django.db import connection
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.db import connection
from .models import Document, Chunk
import os
from openai import OpenAI
from django.conf import settings
from pgvector.django import CosineDistance
import cohere
import json
import boto3
from uuid import uuid4
from io import BytesIO

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def detect_section(chunk):
    lower = chunk.lower()

    if "technical skills" in lower or "programming languages" in lower or "skills" in lower:
        return "Skills"

    if "professional experience" in lower or "experience" in lower or "developer" in lower:
        return "Experience"

    if "projects" in lower or "retrieval-augmented" in lower or "rag" in lower:
        return "Projects"

    if "education" in lower or "university" in lower or "bachelor" in lower:
        return "Education"

    return "Other"


def retrieve_relevant_chunks(query, document_id, retrieve_k=15, rerank_k=3):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
    except Exception as e:
        print("EMBEDDING ERROR:", repr(e))
        raise

    query_embedding = response.data[0].embedding

    chunks = list(
        Chunk.objects
        .filter(document_id=document_id)
        .exclude(embedding=None)
        .order_by(CosineDistance("embedding", query_embedding))[:retrieve_k]
    )

    if not chunks:
        return {
            "top_chunks": [],
            "sources": [],
            "confidence": "Low",
            "top_rerank_score": 0,
        }

    documents = [chunk.chunk_text for chunk in chunks]

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=min(rerank_k, len(documents))
    )
    
    if not rerank_response.results:
        return {
            "top_chunks": [],
            "confidence": "Low",
            "top_rerank_score": 0,
            "sources": []
        }

    best_match = rerank_response.results[0]
    rerank_score = best_match.relevance_score

    if rerank_score >= 0.20:
        confidence = "High"
    elif rerank_score >= 0.10:
        confidence = "Medium"
    else:
        confidence = "Low"

    top_chunks = [
        chunks[result.index]
        for result in rerank_response.results
    ]

    sources = []

    for result in rerank_response.results:
        chunk = chunks[result.index]

        sources.append({
            "chunk_id": chunk.id,
            "section": chunk.section,
            "rerank_score": result.relevance_score,
            "text_preview": chunk.chunk_text[:300],
        })

    return {
        "top_chunks": top_chunks,
        "sources": sources,
        "confidence": confidence,
        "top_rerank_score": rerank_score,
    }


def analyze_resume_match(document, job_description):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    retrieval = retrieve_relevant_chunks(
        query=job_description,
        document_id=document.id,
        retrieve_k=15,
        rerank_k=5
    )

    resume_context = "\n\n".join(
        chunk.chunk_text for chunk in retrieval["top_chunks"]
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
            "role": "system",
            "content": """
                You are an AI recruiting assistant.

                Evaluate ONLY how well the resume context matches the provided job description.

                Use ONLY the provided resume context.
                Do NOT invent requirements that are not in the job description.
                Do NOT mention missing skills unless they appear in the job description.
                Do NOT penalize the candidate for lacking tools not listed in the job description.

                Return valid JSON only with this exact structure:
                {
                "match_score": 0,
                "strengths": [],
                "missing_skills": [],
                "recommendation": ""
                }

                Scoring rules:
                - match_score must be an integer from 0 to 100.
                - 90-100 = excellent match with most required skills present
                - 75-89 = strong match with several required skills present
                - 60-74 = moderate match with some required skills present
                - 40-59 = weak match with few required skills present
                - 0-39 = poor match with little relevant evidence

                For strengths:
                - Include only skills or experience found in the resume context.
                - Prefer exact matches to the job description.

                For missing_skills:
                - Include only skills from the job description that are not found in the resume context.

                If resume context is weak or irrelevant, use a lower score.
                """
            },
            {
            "role": "user",
            "content": f"""
                Job Description:
                {job_description}

                Resume Context:
                {resume_context}
                """
            }
        ]
    )

    analysis = json.loads(response.choices[0].message.content)

    strengths = analysis.get("strengths", [])
    missing_skills = analysis.get("missing_skills", [])

    matched_count = len(strengths)
    missing_count = len(missing_skills)
    total_required = matched_count + missing_count

    if total_required > 0:
        deterministic_score = round((matched_count / total_required) * 100)
    else:
        deterministic_score = 0

    analysis["match_score"] = deterministic_score

    return {
        "document_id": document.id,
        "filename": document.filename,
        "analysis": analysis,
        "confidence": retrieval["confidence"],
        "top_rerank_score": retrieval["top_rerank_score"],
        "sources": retrieval["sources"],
    }
        

@api_view(["GET"])
def health_check(request):
    return Response({"status": "ok"})


@api_view(["GET"])
def home(request):
    return Response({
        "service": "JobFit AI",
        "status": "running"
    })


@api_view(["POST"])
def upload_resume(request):
    uploaded_file = request.FILES.get("file")

    if not uploaded_file:
        return Response({"error": "No file uploaded"}, status=400)

    pdf_bytes = uploaded_file.read()

    reader = PdfReader(BytesIO(pdf_bytes))

    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    s3 = boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME,
    )

    s3_key = f"resumes/{uuid4()}_{uploaded_file.name}"

    s3.upload_fileobj(
        BytesIO(pdf_bytes),
        settings.AWS_STORAGE_BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "application/pdf"}
    )

    s3_url = (
        f"https://{settings.AWS_STORAGE_BUCKET_NAME}"
        f".s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{s3_key}"
    )
    
    document = Document.objects.create(
        filename=uploaded_file.name,
        document_type="resume",
        s3_key=s3_key,
        s3_url=s3_url
    )

    chunks = chunk_text(text)

    for index, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)

        Chunk.objects.create(
            document=document,
            chunk_index=index,
            chunk_text=chunk,
            section=detect_section(chunk),
            embedding=embedding
        )

    return Response({
        "document_id": document.id,
        "filename": document.filename,
        "s3_key": document.s3_key,
        "s3_url": document.s3_url,
        "text_length": len(text),
        "chunks_created": len(chunks)
    })


@api_view(["POST"])
def search_resume(request):
    query = request.data.get("query")

    if not query:
        return Response({"error": "Missing query"}, status=400)

    query_embedding = create_embedding(query)

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, chunk_text
            FROM rag_chunk
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT 5;
            """,
            [query_embedding],
        )

        rows = cursor.fetchall()

    return Response({
        "query": query,
        "results": [
            {
                "chunk_id": row[0],
                "chunk_text": row[1]
            }
            for row in rows
        ]
    })
    

def run_rag_pipeline(query, document_id):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    retrieval = retrieve_relevant_chunks(
        query=query,
        document_id=document_id,
        retrieve_k=15,
        rerank_k=3
    )

    top_chunks = retrieval["top_chunks"]

    context = "\n\n".join(
        chunk.chunk_text
        for chunk in top_chunks
    )

    answer = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a resume question-answering assistant.

                    Answer the user's question directly in 1-3 sentences.

                    Use only the provided resume context.

                    Mention the specific skill, tool, project, or experience from the context when possible.

                    If the context does not contain the answer, say:
                    "I cannot determine this from the resume."
                    """
            },
            {
                "role": "user",
                "content": f"Resume Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]
    )

    return {
        "answer": answer.choices[0].message.content,
        "contexts": [chunk.chunk_text for chunk in top_chunks],
        "chunk_ids": [chunk.id for chunk in top_chunks],
        "sources": retrieval["sources"],
        "confidence": retrieval["confidence"],
        "top_rerank_score": retrieval["top_rerank_score"],
    } 
    
    
@api_view(["POST"])
def ask_resume(request):
    query = request.data.get("query")
    document_id = request.data.get("document_id")

    if not query:
        return Response({"error": "Missing query"}, status=400)

    if not document_id:
        return Response({"error": "Missing document_id"}, status=400)

    result = run_rag_pipeline(query, document_id)

    return Response({
        "query": query,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "top_rerank_score": result["top_rerank_score"],
        "sources": result["sources"],
    })
    

@api_view(["POST"])
def match_job(request):
    document_id = request.data.get("document_id")
    job_description = request.data.get("job_description")

    if not document_id:
        return Response({"error": "Missing document_id"}, status=400)

    if not job_description:
        return Response({"error": "Missing job_description"}, status=400)

    try:
        document = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        return Response({"error": "Document not found"}, status=404)

    result = analyze_resume_match(document, job_description)

    return Response({
        "job_description": job_description,
        **result
    })


@api_view(["POST"])
def rank_candidates(request):
    job_description = request.data.get("job_description")
    document_ids = request.data.get("document_ids")

    if not job_description:
        return Response({"error": "Missing job_description"}, status=400)

    if document_ids:
        documents = Document.objects.filter(id__in=document_ids)
    else:
        documents = Document.objects.filter(document_type="resume")

    results = []

    for document in documents:
        result = analyze_resume_match(document, job_description)

        match_score = result["analysis"].get("match_score", 0)

        results.append({
            "document_id": result["document_id"],
            "filename": result["filename"],
            "match_score": match_score,
            "strengths": result["analysis"].get("strengths", []),
            "missing_skills": result["analysis"].get("missing_skills", []),
            "recommendation": result["analysis"].get("recommendation", ""),
            "confidence": result["confidence"],
            "top_rerank_score": result["top_rerank_score"],
        })

    results = sorted(
        results,
        key=lambda x: x["match_score"],
        reverse=True
    )
    for index, candidate in enumerate(results, start=1):
        candidate["rank"] = index
    return Response({
        "job_description": job_description,
        "candidate_count": len(results),
        "candidates": results
    })
        

@api_view(["GET"])
def list_documents(request):
    documents = Document.objects.all().order_by("-created_at")

    return Response({
        "documents": [
            {
                "id": document.id,
                "filename": document.filename,
                "document_type": document.document_type,
                "created_at": document.created_at
            }
            for document in documents
        ]
    })
    

@api_view(["DELETE"])
def delete_document(request, document_id):
    try:
        document = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        return Response({"error": "Document not found"}, status=404)

    filename = document.filename
    document.delete()

    return Response({
        "message": "Document deleted",
        "document_id": document_id,
        "filename": filename
    })
    