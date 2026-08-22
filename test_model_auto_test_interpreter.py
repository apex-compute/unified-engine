#!/usr/bin/env python3
"""Focused tests for per-model interpreter selection in model_auto_test.py."""

import os
import sys
import unittest
from unittest import mock

import model_auto_test


class ModelInterpreterSelectionTest(unittest.TestCase):
    def test_models_without_override_use_harness_interpreter(self):
        self.assertEqual(
            model_auto_test._python_executable_for_test({"name": "plain"}),
            sys.executable,
        )

    def test_configured_environment_override_is_used(self):
        with mock.patch.dict(os.environ, {"TEST_MODEL_PYTHON": "/opt/model/bin/python"}):
            self.assertEqual(
                model_auto_test._python_executable_for_test(
                    {"name": "isolated", "python_env": "TEST_MODEL_PYTHON"}
                ),
                "/opt/model/bin/python",
            )

    def test_empty_environment_override_falls_back(self):
        with mock.patch.dict(os.environ, {"TEST_MODEL_PYTHON": ""}):
            self.assertEqual(
                model_auto_test._python_executable_for_test(
                    {"name": "isolated", "python_env": "TEST_MODEL_PYTHON"}
                ),
                sys.executable,
            )

    def test_all_gemma4_registry_entries_use_isolated_environment(self):
        gemma4_tests = [test for test in model_auto_test.TESTS
                        if test["name"].startswith("gemma4_")]
        self.assertTrue(gemma4_tests)
        self.assertEqual(
            {test.get("python_env") for test in gemma4_tests},
            {"GEMMA4_PYTHON"},
        )


if __name__ == "__main__":
    unittest.main()
