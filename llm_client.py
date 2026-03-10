

import os
import httpx

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"

async def call_llm(system_prompt, user_prompt, max_tokens=800, model=DEFAULT_MODEL):
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠ GROQ_API_KEY not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        return f"⚠ API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return f"⚠ Error: {str(e)}"

# Old name still works
call_gemini = call_llm