# aws-rag-bot
A simple RAG application to test some AWS features

## Setup

### Environment variables

1. Create your local env file:
	- Copy `.env.example` to `.env`
2. Fill at minimum:
	- `S3_BUCKET_NAME`

AWS authentication can be provided either via `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, or via your normal AWS configuration (profiles/SSO), depending on how you run the app.

### Run locally

```bash
uvicorn app.main:app --reload
```

## API

- `GET /` health check
- `GET /s3/files?prefix=...` list files in the configured bucket
- `POST /s3/upload` upload a file (multipart field: `file`, optional query: `key`)
- `DELETE /s3/files/{key}` delete a file by key
