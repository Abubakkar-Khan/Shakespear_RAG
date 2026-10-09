from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

print(f"API Key (first 10 chars): {GROQ_API_KEY[:10]}...")
print(f"Target Model: {model}")

client = Groq(api_key=GROQ_API_KEY)

try:
    print(f"\nAttempting completion with '{model}'...")
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello in 5 words"}],
        model=model,
        temperature=0.7,
        max_tokens=100
    )
    print(f"\nSuccess with {model}! Response:")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print(f"\nNote: Primary model '{model}' unavailable on Groq: {e}")
    # Official Groq replacement for Llama 3.1 8B Instant
    fallback_model = "openai/gpt-oss-20b"
    print(f"Falling back to Groq replacement model: {fallback_model}...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello in 5 words"}],
            model=fallback_model,
            temperature=0.7,
            max_tokens=1024
        )
        msg = chat_completion.choices[0].message
        ans = msg.content or getattr(msg, 'reasoning', '')
        print(f"\nSuccess with fallback ({fallback_model})! Response:")
        print(ans)
    except Exception as fallback_err:
        print(f"\nFallback error: {fallback_err}")
        import traceback
        traceback.print_exc()

