from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status

from app.routes.s3 import router as s3_router
from app.routes.text import router as text_router
from app.services.dependencies import get_sagemaker_docs_sync_service
from app.services.s3_service import S3ServiceError


def _ensure_logging() -> None:
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        root.setLevel(logging.INFO)
        for handler in root.handlers:
            handler.setFormatter(formatter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_logging()
    await get_sagemaker_docs_sync_service().startup_check_and_sync_docs()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(s3_router)
app.include_router(text_router)


@app.exception_handler(S3ServiceError)
async def s3_service_error_handler(request: Request, exc: S3ServiceError) -> JSONResponse:
    """Map S3 service-layer failures to a consistent HTTP response.

    This keeps AWS/S3 errors from leaking internal details to API consumers while still
    returning a predictable payload the frontend/clients can handle.

    Returns:
        502 Bad Gateway with a JSON body: {"detail": "..."}
    """
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


@app.get("/")
async def root():
    return {"message": "Hello World! AWS RAG Bot is running."}
