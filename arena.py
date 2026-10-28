import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)


def test_model_behavior(question, temperature):
    """Test model behavior with the same question at different temperatures."""
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        extra_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Behavior Test",
        },
    )
    return response.choices[0].message.content


# Example question to test model behavior
default_question = "Write a 2-sentence story about a toaster that gains consciousness, but end every sentence with a random rhyming word."

# Test at temperature 0 (deterministic)
response_t0 = test_model_behavior(default_question, 0)
print(f"Temperature 0 response:\n{response_t0}\n")

# Test at temperature 2 (creative)
response_t2 = test_model_behavior(default_question, 1.5)
print(f"Temperature 2 response:\n{response_t2}\n")
