import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, List,Dict
from pydantic import BaseModel

import uvicorn, shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from agent import TranscriberAgent
from DoBanglaSummarize import DoBanglaSummarize

agent = TranscriberAgent()
summarizer_agent = DoBanglaSummarize()

class Conversation(BaseModel):
    data: List[List[str]]

app = FastAPI()
origins = ["*"]
CORS_ALLOW_HEADERS = ['*']
CORS_ORIGIN_ALLOW_ALL = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome."}

@app.post("/transcribe")
def transcribe(file: UploadFile = File(...), language: Optional[str] = Form(None)) -> Any:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = Path(temp_file.name)

    print(f"File saved at {temp_file_path}")
    if not language:
        language = 'bn'

    text = agent.get_raw_transcription(str(temp_file_path),language=language)
    print(text)
    
    os.unlink(temp_file_path)

    return {
        "content": {
            "filename": file.filename,
            "transcription": text,
        }
    }

@app.post("/converse")
def converse(file: UploadFile = File(...), language: Optional[str] = Form(None)) -> Any:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = Path(temp_file.name)
    
    if not language:
        language = 'bn'
    print(f"File saved at {temp_file_path}")

    conversation = agent.create_conversation(str(temp_file_path),language=language)
    print(conversation)    

    os.unlink(temp_file_path)

    return {
        "content": {
            "filename": file.filename,
            "transcription": conversation,
        }
    }
    
@app.post("/summarize")
def summarize(file: UploadFile = File(...), language: Optional[str] = Form(None)) -> Any:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = Path(temp_file.name)

    if not language:
        language = 'bn'
    print(f"File saved at {temp_file_path}")

    transcription = agent.get_raw_transcription(str(temp_file_path),language=language)
    summary = summarizer_agent.summary_kore_felo(transcription,language=language)
    
    os.unlink(temp_file_path)

    return {
        "content": {
            "filename": file.filename,
            "summary": summary,
        }
    }
    
@app.post("/get_keyword")
def keyword_extract(file: UploadFile = File(...), keyword_str:  Optional[str] = Form(None), language: Optional[str] = Form(None)) -> Any:

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = Path(temp_file.name)

    if not language:
        language = 'bn'

    print(f"File saved at {temp_file_path}")
    
    if keyword_str:
        keyword_list = keyword_str.split(",")
        print(f'Got Keys: {keyword_list}')
        keyword = agent.get_keywords(str(temp_file_path), keywords=keyword_list, language=language)
    else:
        keyword = agent.get_keywords(str(temp_file_path),language=language)
    print(f'Got Response: {keyword}')
    os.unlink(temp_file_path)
    
    return {
        "content": {
            "filename": file.filename,
            "keywords": keyword,
        }
     }
    
    
if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=9803)