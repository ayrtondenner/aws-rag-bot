from fastapi import FastAPI

from routes.s3 import router as s3_router

app = FastAPI()

app.include_router(s3_router)


@app.get("/")
async def root():
    return {"message": "Hello World! AWS RAG Bot is running."}