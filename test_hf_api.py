import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="fireworks-ai",
    api_key = os.getenv("HF_API_KEY")
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-0528",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)