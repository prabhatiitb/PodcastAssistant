import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Try .env first (local), fall back to Streamlit secrets (cloud deployment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env or Streamlit secrets")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.6-flash"  # current stable flash model
