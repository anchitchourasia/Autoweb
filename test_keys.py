import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 55)
print("🔑  AutoWeb — API Key Tester")
print("=" * 55)

# ── Test 1: Gemini ─────────────────────────────────────────
print("\n📌 Testing GEMINI (google-genai)...")
try:
    from google import genai
    key = os.getenv("GEMINI_API_KEY")
    assert key, "GEMINI_API_KEY not found in .env"
    client = genai.Client(api_key=key)
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one word."
    )
    print(f"✅ Gemini works! → {r.text.strip()}")
except Exception as e:
    print(f"❌ Gemini failed: {e}")

# ── Test 2: Groq ────────────────────────────────────────────
print("\n📌 Testing GROQ...")
try:
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    assert key, "GROQ_API_KEY not found in .env"
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10
    )
    text = response.choices[0].message.content
    print(f"✅ Groq works! → {text.strip()}")
except Exception as e:
    print(f"❌ Groq failed: {e}")

# ── Test 3: Ollama Cloud ────────────────────────────────────
print("\n📌 Testing OLLAMA CLOUD...")
try:
    from ollama import Client
    key = os.getenv("OLLAMA_API_KEY")
    assert key, "OLLAMA_API_KEY not found in .env"
    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {key}"}
    )
    response = client.chat(
        model="gpt-oss:120b-cloud",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        stream=False
    )
    print(f"✅ Ollama Cloud works! → {response['message']['content'].strip()}")
except Exception as e:
    print(f"❌ Ollama Cloud failed: {e}")

print("\n" + "=" * 55)
print("Done! Fix any ❌ before running the app.")
print("=" * 55)