import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into environment

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
