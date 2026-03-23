from __future__ import annotations

from src.utils import logger as logger_mod


def test_logger_get_logger():
    log = logger_mod.get_logger("tests.sample")
    assert log.name == "tests.sample"
