# Lab Submission Checklist

Use this checklist to prepare and verify your submission for BBT 4206 - CAT 1.

## Team & Contribution Details
- [ ] Team name (GitHub Classroom):
- [ ] Member 1: Student ID, Name, contribution (branch link)
- [ ] Member 2: Student ID, Name, contribution (branch link)
- [ ] Member 3: Student ID, Name, contribution (branch link)
- [ ] Member 4: Student ID, Name, contribution (branch link)
- [ ] Member 5: Student ID, Name, contribution (branch link)

## Administrative
- [ ] Chosen level of difficulty: baseline / intermediate / advanced
- [ ] Video demo (<= 5 minutes) — provide shareable link (do NOT upload to repo)
- [ ] Public app URL (Gradio or Streamlit):

## Core repository artifacts
- [ ] Updated `api.py` endpoints to serve models (see file: api.py)
  - [ ] Baseline: at least 3 models served
  - [ ] Intermediate (recommended): Naive Bayes, kNN, SVM, Random Forest
  - [ ] Advanced (optional): cluster-classifier endpoint
- [x] Association-rules recommender endpoint (rules loaded from disk)
- [x] Cluster classifier endpoint (if choosing advanced)

## Web / Demo
- [ ] Public-hosted model app URL (Hugging Face Spaces Gradio or Streamlit Community Cloud)
- [ ] (Optional) Simple HTML/CSS/JS demo page(s) demonstrating API usage

## Error handling & robustness (recommended)
- [ ] Basic input validation implemented for API endpoints
- [ ] Clear error messages for missing/invalid inputs

## Docker & Deployment (advanced)
- [ ] `Dockerfile` for Flask + Gunicorn created
- [ ] (Optional) `docker-compose.yml` or Nginx reverse-proxy setup
- [ ] Build and run instructions added to `README.md`

## Submission package / README
- [ ] `README.md` updated with run/deploy instructions and example requests
- [ ] Sample curl / JavaScript examples for each important endpoint
- [ ] Branch link(s) used for implementation (for each team member where applicable)

## Final checks before submission
- [ ] Ensure video link is accessible to lecturer
- [ ] Ensure public app URL is reachable
- [ ] Ensure all required endpoints work locally and are documented

---
Fill the checklist and commit. If you want, I can:
- create or update `api.py` endpoints scaffold
- add example curl requests and a basic `README.md` section
- create a minimal `Dockerfile` and `docker-compose.yml` scaffold

Fill the fields above or tell me which of the optional tasks you want me to do next.
