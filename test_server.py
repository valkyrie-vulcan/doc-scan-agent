# test_server.py
from fastapi import FastAPI, File, UploadFile
from typing import Optional

app = FastAPI()

@app.post("/upload-test/")
async def create_upload_file(image: UploadFile):
    if not image:
        return {"error": "No file sent"}
    return {"filename": image.filename, "content_type": image.content_type}

@app.get("/")
def read_root():
    return {"Hello": "World"}