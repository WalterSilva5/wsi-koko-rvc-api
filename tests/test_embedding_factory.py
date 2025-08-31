"""
Testes unitários para Embedding Factory
"""

import pytest
from project.embedding.factory import Factory


def test_create_embedding_returns_expected_type():
    factory = Factory()
    result = factory.create_embedding()
    assert result is not None
