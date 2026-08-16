import os
import json

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Administrative Assistant"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GEMINI
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY is not configured.")

client = None

if API_KEY:
    client = genai.Client(
        api_key=API_KEY
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class EmailRequest(BaseModel):

    email: str

    question: str = ""


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        "index.html"
    )


# ============================================================
# AI EMAIL ANALYSIS
# ============================================================

@app.post("/api/analyze")
def analyze_email(request: EmailRequest):

    email = request.email.strip()

    question = request.question.strip()


    if not email:

        return {
            "error": "Please provide an email."
        }


    if not client:

        return {
            "error": "Gemini API key is not configured."
        }


    prompt = f"""
You are an AI Administrative Assistant.

Analyze the following email.

Return ONLY valid JSON.

Required JSON structure:

{{
    "summary": "...",
    "category": "...",
    "priority": "...",
    "deadline": "...",
    "actions": "...",
    "reply": "...",
    "answer": "..."
}}

CATEGORY must be one of:

HR
Finance
IT Support
Meeting
Operations
Marketing
Legal
General
Spam

PRIORITY must be one of:

High
Medium
Low

Rules:

1. Summarize the email clearly.
2. Identify the correct category.
3. Detect urgency.
4. Extract meeting dates, meeting times and deadlines.
5. Extract action items.
6. Mention responsible persons when available.
7. Write a professional email reply.
8. If a question is provided, answer it using the email.
9. If no question is provided, set answer to "No question asked."
10. Do not invent information.

EMAIL:

{email}

USER QUESTION:

{question if question else "No question asked."}
"""


    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )


        text = response.text.strip()


        # Remove possible Markdown JSON fences

        if text.startswith("```"):

            text = text.replace(
                "```json",
                ""
            )

            text = text.replace(
                "```",
                ""
            )

            text = text.strip()


        data = json.loads(text)


        return data


    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "running",
        "gemini": bool(API_KEY)
    }