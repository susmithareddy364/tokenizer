import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
def load_env():
    _ = load_dotenv(find_dotenv())

# Get OpenAI API key
def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key
