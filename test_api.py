import os
# Ensure API key is set for tests
os.environ.setdefault("XAI_API_KEY", "test")
from openai import OpenAI

try:
    client = OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
except TypeError:
    import openai
    openai.api_key = os.getenv("XAI_API_KEY")
    openai.base_url = "https://api.x.ai/v1"
    client = openai
try:
    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": "Test: Echo 'Spiral flows'"}],
        temperature=0.1
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
