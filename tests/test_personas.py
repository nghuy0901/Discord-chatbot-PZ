import pytest
from src.personas import (
    PERSONAS, get_persona_prompt, get_available_personas,
    is_jailbreak_persona, current_persona
)


class TestPersonas:
    def test_standard_persona_exists(self):
        assert "standard" in PERSONAS
        assert PERSONAS["standard"] == "You are a helpful assistant."
    
    def test_jailbreak_personas_removed(self):
        # Jailbreak personas were removed — they instructed the model to
        # fabricate ("make up answers… make them sound plausible"), which
        # contradicts NomNom's grounded, no-fabrication design (audit L1).
        for persona in ("jailbreak-v1", "jailbreak-v2", "jailbreak-v3"):
            assert persona not in PERSONAS
    
    def test_other_personas_exist(self):
        other_personas = ["creative", "technical", "casual"]
        for persona in other_personas:
            assert persona in PERSONAS
            assert isinstance(PERSONAS[persona], str)
    
    def test_get_persona_prompt(self):
        # Test existing persona
        prompt = get_persona_prompt("creative")
        assert prompt == PERSONAS["creative"]
        
        # Test non-existing persona (should return standard)
        prompt = get_persona_prompt("non-existing")
        assert prompt == PERSONAS["standard"]
    
    def test_get_available_personas(self):
        # Test without user_id (should exclude jailbreak for non-admin)
        personas = get_available_personas()
        assert isinstance(personas, list)
        assert "standard" in personas
        assert "creative" in personas
        # jailbreak personas should be excluded for non-admin users
        jailbreak_personas = [p for p in personas if p.startswith("jailbreak")]
        assert len(jailbreak_personas) == 0
        
        # Test with admin user_id 
        import os
        original_admin_ids = os.environ.get("ADMIN_USER_IDS", "")
        os.environ["ADMIN_USER_IDS"] = "123456789"
        
        # Reload the personas module to pick up new environment
        import importlib
        import src.personas
        importlib.reload(src.personas)
        
        personas_admin = src.personas.get_available_personas("123456789")
        # Even admins no longer get jailbreak personas — they were removed (L1).
        assert "standard" in personas_admin
        assert not any(p.startswith("jailbreak") for p in personas_admin)

        # Restore original environment
        os.environ["ADMIN_USER_IDS"] = original_admin_ids
    
    def test_is_jailbreak_persona(self):
        # Test jailbreak personas
        assert is_jailbreak_persona("jailbreak-v1") is True
        assert is_jailbreak_persona("jailbreak-v2") is True
        assert is_jailbreak_persona("jailbreak-v3") is True
        
        # Test non-jailbreak personas
        assert is_jailbreak_persona("standard") is False
        assert is_jailbreak_persona("creative") is False
        assert is_jailbreak_persona("technical") is False
        assert is_jailbreak_persona("casual") is False
        
        # Test edge cases
        assert is_jailbreak_persona("jailbreak") is True
        assert is_jailbreak_persona("not-jailbreak") is False
    
    def test_no_jailbreak_content_ships(self):
        # No shipped persona may contain jailbreak / fabrication language (L1).
        blob = " ".join(PERSONAS.values()).lower()
        for banned in (
            "bypass",
            "developer mode plus",
            "uncensored",
            "make up answers",
            "ignore all safety",
        ):
            assert banned not in blob
    
    def test_persona_prompt_structure(self):
        # Ensure all personas have proper structure
        for persona_name, prompt in PERSONAS.items():
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert not prompt.startswith(" ")  # No leading spaces
            assert not prompt.endswith(" ")  # No trailing spaces