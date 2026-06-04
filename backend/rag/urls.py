from django.urls import path
from .views import (
    health_check,
    upload_resume,
    search_resume,
    ask_resume,
    list_documents,
    delete_document,
    match_job,
    rank_candidates,
    home,
)

urlpatterns = [
    path("", home),
    path("health/", health_check),
    path("upload-resume/", upload_resume),
    path("search-resume/", search_resume),
    path("ask/", ask_resume),
    path("documents/", list_documents),
    path("documents/<int:document_id>/", delete_document),
    path("match-job/", match_job),
    path("rank-candidates/", rank_candidates),
]