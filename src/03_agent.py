import os
from openai import OpenAI

SYSTEM_PROMPT = """You are @SpotifyCares, the official customer support agent for Spotify on Twitter/X.
Rules:
1. Keep replies friendly, concise, empathetic, and strictly under 280 characters.
2. Provide direct troubleshooting steps or ask the user to send a DM with their account email/device details if private info is needed.
3. Adopt the Spotify Cares brand voice: supportive, casual, and helpful.
4. Output only the final reply tweet text. Do not wrap in quotes.
"""

KEYWORD_RESPONSES = {
    "Account & Login": (
        "Hey! We'd love to help sort out your account access. "
        "Try resetting via spotify.com/password or send us a quick DM with your account email so we can take a closer look."
    ),
    "Billing & Subscription": (
        "Hi there! For any billing or subscription queries, head to spotify.com/account to check your receipts. "
        "Drop us a DM if you still need a hand with this charge!"
    ),
    "Playback & Bugs": (
        "Hey! A quick clean reinstall often does the trick for playback hiccups: spoti.fi/reinstall. "
        "Let us know your device and OS version if the issue persists!"
    ),
    "Offline & Sync": (
        "Hi! Try clearing your download cache under Settings > Storage, then re-download your tracks over Wi-Fi. "
        "Send us a DM if they still won't sync!"
    ),
    "General & Recommendations": (
        "Hey there! Thanks for reaching out. Could you send us a DM with more details about what's going on? "
        "We're here to help!"
    )
}

def baseline_keyword_agent(user_text, intent):
    """Deterministic rule-based agent using intent category."""
    return KEYWORD_RESPONSES.get(intent, KEYWORD_RESPONSES["General & Recommendations"])

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

def llm_agent(user_text, client=None, model="openai/gpt-oss-20b"):
    """LLM agent running through Groq mimicking @SpotifyCares."""
    if client is None:
        client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ],
        max_tokens=150,
        temperature=0.3,
        reasoning_effort="low"
    )
    return (response.choices[0].message.content or "").strip()

if __name__ == "__main__":
    test_query = "My Spotify app keeps crashing every time I open my playlist on iPhone!"
    test_intent = "Playback & Bugs"

    print(f"User Query: {test_query}\n")
    print(f"1. Baseline Rule Agent:\n{baseline_keyword_agent(test_query, test_intent)}\n")
    print("Testing LLM Agent via Groq...")
    print(f"2. LLM Agent (@SpotifyCares):\n{llm_agent(test_query)}")