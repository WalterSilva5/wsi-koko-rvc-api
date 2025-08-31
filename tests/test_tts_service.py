"""
Testes unitários para TTSService
"""

import pytest
from project.tts.tts_service import TTSService


def test_synthesize_returns_expected_type():
    tts = TTSService()
    result = tts.synthesize()
    assert result is not None
