"""
Testes unitários para Application
"""

import pytest
from project.core.application import Application


def test_run_does_not_raise():
    app = Application()
    try:
        app.run()
    except Exception:
        pytest.fail("Application.run() levantou exceção")
