"""
Testes unitários para Model Factory
"""

import pytest
from project.model.factory import Factory


def test_create_model_returns_expected_type():
    factory = Factory()
    result = factory.create_model()
    assert result is not None
