"""
Testes unitários para ConversorService
"""

import pytest
from project.conversor.service import Service


def test_convert_returns_expected_type():
    service = Service()
    # Supondo que convert() retorna um dict ou similar
    result = service.convert()
    assert isinstance(result, dict) or result is None
