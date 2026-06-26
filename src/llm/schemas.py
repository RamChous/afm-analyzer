"""src/llm/schemas.py
JSON-схемы для structured output Ollama. Описывают ТОЛЬКО то, что
генерит модель на каждом проходе. severity тут намеренно нет —
он проставляется кодом из Rule.
"""

SPEAKER_ROLES_SCHEMA = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "role": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["speaker", "role"],
            },
        }
    },
    "required": ["roles"],
}

VIOLATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "timestamp": {"type": "number"},
                    "quote": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["rule_id", "timestamp", "quote", "reasoning", "confidence"],
            },
        }
    },
    "required": ["violations"],
}