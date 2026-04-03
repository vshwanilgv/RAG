# save as test_setup.py in your project root
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
if not key:
    print("ERROR: OPENAI_API_KEY not found. Check your .env file.")
else:
    print(f"Key loaded: {key[:8]}...{key[-4:]}")  # shows partial key only

# Test a real API call
client = OpenAI(api_key=key)
response = client.embeddings.create(
    input="test financial document",
    model="text-embedding-3-small"
)
print(f"Embedding dimension: {len(response.data[0].embedding)}")
print("OpenAI connection working.")