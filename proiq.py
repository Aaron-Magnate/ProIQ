
try:
    import sqlite3
except ImportError:
    import pysqlite3 as sqlite3
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import os
import io
import openai
from openai import OpenAI
import base64
from app.services.user_authorization import utils as user_utils
import logging
import secrets
from pathlib import Path
# from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from app.models.response import StandardResponse, HTTPStatusCodeEnum

# --- NEW/UPDATED IMPORTS ---
from sqlalchemy.orm import Session
from app.database import get_db # Your dependency to get a DB session
# from app.core.security import get_current_user, get_current_user_google_addon, get_current_user_from_refresh_token
# Import both User and Team models
from app.models.user import User

from fastapi import Query
# from app.core.security import create_access_token, get_user_by_email, create_refresh_token
from groq import Groq
import logging

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)
router = APIRouter()

DB_FILE = "proiq.db"
FILE_STORAGE_PATH = "stored_files"

# Create the base directory for storing files if it doesn't exist
os.makedirs(FILE_STORAGE_PATH, exist_ok=True)

def init_db():
    """Initializes the SQLite database for RAG data."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        
        # --- UPDATED `files` table ---
        # Added `permission_level` to control access. Defaults to 'user'.
        # Added `added_by_user_id` and `added_by_user_name` to track who added the file.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,                
                filename TEXT NOT NULL,
                mimetype TEXT NOT NULL,
                filepath TEXT NOT NULL,                
                added_by_user_id INTEGER,
                added_by_user_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP                
            )
        """)

        

        
        conn.commit()

init_db()


@router.post("/upload_file", response_model=StandardResponse)
async def upload_file(
    current_user: User = Depends(user_utils.get_current_user),
    file: UploadFile = File(...),    
    db: Session = Depends(get_db)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    

    """Uploads a file and stores its metadata in the database."""
    try:
        # Save the file to the filesystem
        file_location = os.path.join(FILE_STORAGE_PATH, file.filename)
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Store file metadata in the database
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO files (filename, mimetype, filepath, added_by_user_id, added_by_user_name)
                VALUES (?, ?, ?, ?, ?)
            """, (file.filename, file.content_type, file_location, current_user.id, current_user.fname + " " + current_user.lname))
            conn.commit()
        
        return StandardResponse(
            status="success",
            message="File uploaded successfully.",
            data=[],
            code=HTTPStatusCodeEnum.OK
        )
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    


@router.get("/list_files", response_model=StandardResponse)
def list_files(
    current_user: User = Depends(user_utils.get_current_user),
    db: Session = Depends(get_db)
):
    """Lists all uploaded files."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user_id = current_user.id

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, mimetype, filepath, added_by_user_id, added_by_user_name, created_at FROM files WHERE added_by_user_id = ?",
                (user_id,)
            )
            rows = cursor.fetchall()
            
            files = [
                {
                    "id": row[0],
                    "filename": row[1],
                    "mimetype": row[2],
                    "filepath": row[3],
                    "added_by_user_id": row[4],
                    "added_by_user_name": row[5],
                    "created_at": row[6]
                }
                for row in rows
            ]
        
        return StandardResponse(
            status="success",
            message="Files retrieved successfully.",
            data=files,
            code=HTTPStatusCodeEnum.OK
        )
    except Exception as e:
        logging.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
from fastapi import Path

@router.get("/download_file/{file_id}")
def download_file(
    current_user: User = Depends(user_utils.get_current_user),
    file_id: int = Path(..., description="The ID of the file to download")
):
    """Downloads a file by its ID."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user_id = current_user.id

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT filename, filepath, mimetype, added_by_user_id FROM files WHERE id = ?",
                (file_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, filepath, mimetype, added_by_user_id = row
            
            # Ensure the user is authorized to download the file
            if added_by_user_id != user_id:
                raise HTTPException(status_code=403, detail="Forbidden")
            
            if not os.path.exists(filepath):
                raise HTTPException(status_code=404, detail="File not found on disk")
            
            # --- 💡 FIX IS HERE ---
            # Change the content disposition type to 'inline' to prevent
            # the browser from automatically triggering a download prompt.
            return FileResponse(
                path=filepath,
                media_type=mimetype,
                filename=filename,
                content_disposition_type="inline" # Add this line
            )
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

@router.delete("/delete_file/{file_id}", response_model=StandardResponse)
def delete_file(
    current_user: User = Depends(user_utils.get_current_user),
    file_id: int = Path(..., description="The ID of the file to delete"),
    db: Session = Depends(get_db)
):
    """Deletes a file by its ID."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user_id = current_user.id

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT filepath, added_by_user_id FROM files WHERE id = ?",
                (file_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="File not found")
            
            filepath, added_by_user_id = row
            
            # Ensure the user is authorized to delete the file
            if added_by_user_id != user_id:
                raise HTTPException(status_code=403, detail="Forbidden")
            
            # Delete the file from the filesystem
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Remove the file record from the database
            cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
            conn.commit()
        
        return StandardResponse(
            status="success",
            message="File deleted successfully.",
            data=[],
            code=HTTPStatusCodeEnum.OK
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    


    