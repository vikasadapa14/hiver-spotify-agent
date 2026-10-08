import os
import json
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

from openai import OpenAI

SYSTEM_PROMPT = """You are @SpotifyCares, the official automated customer support agent for Spotify on Twitter/X.

Your job is to analyze incoming customer tweets and return a strict JSON response.

Intents to classify into:
- "Account & Login" (hacked account, password reset, login verification)
- "Billing & Subscription" (unauthorized charges, student discounts, family plan billing, receipts)
- "Playback & Bugs" (app crashing, songs skipping, bluetooth disconnects, playlist bugs)
- "Offline & Sync" (downloaded tracks not playing offline, local files sync)
- "General & Recommendations" (curation, feedback, feature requests, generic praise/criticism)

Routing & Escalation Policy:
1. Escalate to human ("escalate") if:
   - Account security compromised / hacked / email changed without permission
   - Complex billing/refund disputes requiring sensitive PII (payment receipts, card numbers)
   - User expresses extreme frustration or threatens legal/social escalation
2. Auto-reply directly ("auto_reply") if:
   - Standard troubleshooting applies (clean reinstall, cache clearing, known Spotify links)
   - Asking user for non-sensitive diagnostics (device model, OS version, app version)

Output Format:
You MUST respond ONLY with a valid JSON object matching this schema:
{
  "intent": "<One of the 5 intents above>",
  "decision": "auto_reply" or "escalate",
  "escalation_reason": "<1 concise sentence explaining why, or null if auto_reply>",
  "reply": "<Friendly, empathetic @SpotifyCares tweet reply. Max 280 characters. Include spoti.fi link or ask for device/OS or DM when appropriate. Sign off with initials like /CB or /SP.>"
}
Do not output markdown code blocks. Output raw JSON only.
"""

# Baseline 1: Trivial Baseline (Generic canned DM response for all inquiries)
def trivial_baseline_agent(user_text):
    return {
        "intent": "General & Recommendations",
        "decision": "escalate",
        "escalation_reason": "Trivial baseline passes all queries to human triage.",
        "reply": "Hey there! Thanks for reaching out. Please send us a quick DM with your account email and more details so we can assist!"
    }

# Baseline 2: Simple Rule/Keyword Baseline (Intent-based canned response)
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

def rule_based_baseline_agent(user_text, intent=None):
    if not intent or intent not in KEYWORD_RESPONSES:
        intent = "General & Recommendations"