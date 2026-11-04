# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Failure and skip semantics of suite-level CTest entries."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_support.grouped_gtest import _classify_result


class GroupedGTestRunnerTest(unittest.TestCase):
    def _classify(self, cases: str, count: int, return_code: int = 0) -> int:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            path.write_text(f'<testsuites tests="{count}"><testsuite name="Example">{cases}</testsuite></testsuites>')
            return _classify_result(return_code, path, "Example")[0]

    def test_pass(self) -> None:
        self.assertEqual(self._classify('<testcase name="pass" classname="Example" status="run"/>', 1), 0)

    def test_failure_is_not_hidden_by_skip(self) -> None:
        cases = (
            '<testcase classname="Example" status="run"><failure message="failure"/></testcase>'
            '<testcase classname="Example" status="run"><skipped/></testcase>'
        )
        self.assertEqual(self._classify(cases, 2), 1)

    def test_all_skip(self) -> None:
        self.assertEqual(self._classify('<testcase classname="Example" status="run"><skipped/></testcase>', 1), 77)

    def test_partial_skip(self) -> None:
        cases = (
            '<testcase classname="Example" status="run"/>'
            '<testcase classname="Example" status="run"><skipped/></testcase>'
        )
        self.assertEqual(self._classify(cases, 2), 0)

    def test_nonzero_exit(self) -> None:
        self.assertEqual(self._classify('<testcase classname="Example" status="run"/>', 1, 1), 1)

    def test_empty_or_wrong_suite(self) -> None:
        self.assertEqual(self._classify("", 0), 1)
        self.assertEqual(self._classify('<testcase classname="Other" status="run"/>', 1), 1)

    def test_missing_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(_classify_result(0, Path(directory) / "missing.xml", "Example")[0], 1)

    def test_invalid_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.xml"
            path.write_text("<testsuites")
            self.assertEqual(_classify_result(0, path, "Example")[0], 1)

    def test_root_error_is_not_reported_as_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "error.xml"
            path.write_text(
                '<testsuites tests="1" errors="1"><testsuite name="Example">'
                '<testcase classname="Example" status="run"/></testsuite></testsuites>'
            )
            self.assertEqual(_classify_result(0, path, "Example")[0], 1)


if __name__ == "__main__":
    unittest.main()
