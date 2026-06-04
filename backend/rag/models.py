from django.db import models
from pgvector.django import VectorField


class Document(models.Model):
    filename = models.CharField(max_length=255)
    document_type = models.CharField(max_length=50)
    s3_key = models.CharField(max_length=500, null=True, blank=True)
    s3_url = models.URLField(max_length=1000, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Chunk(models.Model):
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="chunks"
    )

    chunk_index = models.IntegerField()
    chunk_text = models.TextField()

    section = models.CharField(
        max_length=100,
        null=True,
        blank=True
    )

    embedding = VectorField(
        dimensions=1536,
        null=True,
        blank=True
    )