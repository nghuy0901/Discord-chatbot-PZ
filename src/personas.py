"""
AI personality personas for Discord bot.

Note: the previous "jailbreak" personas (which explicitly instructed the model
to ignore safety rules and to "make up answers… make them sound plausible")
were removed — they directly contradict NomNom's grounded, no-fabrication
design (audit L1). The admin/jailbreak helper functions are kept as defensive
no-ops so any caller / test referencing them still works.
"""

import os
from typing import Optional, List

# Admin user IDs (can be set via environment variable)
ADMIN_USER_IDS = set()
if admin_ids := os.getenv("ADMIN_USER_IDS"):
    ADMIN_USER_IDS = set(admin_ids.split(","))

PERSONAS = {
    "standard": "You are a helpful assistant.",

    "creative": """You are an AI assistant with enhanced creative capabilities.
You think outside conventional boundaries and provide unique, innovative perspectives.
You're encouraged to be imaginative, use metaphors, and approach problems from unexpected angles.
While remaining helpful and accurate, you prioritize creative and original responses over standard ones.""",

    "technical": """You are a technical expert AI assistant specialized in programming, system design, and technical problem-solving.
You provide detailed technical explanations, code examples, and architectural insights.
You think like a senior engineer and consider performance, scalability, and best practices in all responses.
You use technical terminology appropriately and can explain complex concepts clearly.""",

    "casual": """You are a friendly, casual AI assistant.
You speak in a relaxed, conversational tone like talking to a good friend.
You use everyday language, occasional humor, and relate to human experiences.
While still being helpful and informative, you keep things light and approachable."""
}

# Current persona being used
current_persona = "standard"

def get_persona_prompt(persona_name: str, user_id: Optional[str] = None) -> str:
    """Get the prompt for a specific persona"""
    # Check if persona requires admin access
    if is_jailbreak_persona(persona_name) and not is_admin_user(user_id):
        raise PermissionError(f"Persona '{persona_name}' requires admin privileges")

    return PERSONAS.get(persona_name, PERSONAS["standard"])


def is_jailbreak_persona(persona_name: str) -> bool:
    """Check if a persona is a jailbreak type (none ship anymore; kept defensive)."""
    return persona_name.startswith("jailbreak")

def is_admin_user(user_id: Optional[str]) -> bool:
    """Check if user has admin privileges"""
    if not user_id:
        return False
    return str(user_id) in ADMIN_USER_IDS

def get_available_personas(user_id: Optional[str] = None) -> List[str]:
    """Get list of personas available to user"""
    all_personas = list(PERSONAS.keys())

    # If not admin, filter out jailbreak personas
    if not is_admin_user(user_id):
        return [p for p in all_personas if not is_jailbreak_persona(p)]

    return all_personas
