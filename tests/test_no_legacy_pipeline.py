"""Архитектурная граница: действия берутся из tool call, а не команд в речи."""

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MARKERS = (
    "NO_CHANGES", "CONVERTER_SYSTEM_PROMPT", "CONVERTER_USER_PROMPT",
    "converter_model", "Ты конвертер", "SET participants.", "SET components.",
)


class NoLegacyPipelineTests(unittest.TestCase):
    def assert_current_runtime(self, code, location):
        for marker in LEGACY_MARKERS:
            self.assertNotIn(marker, code, f"{location}: устаревший протокол {marker!r}")
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                self.assertNotEqual(module, "config.prompts", location)
                self.assertFalse(module == "negotiation_mvp" or module.startswith("negotiation_mvp."), location)
                self.assertNotEqual(module, "langflow_bridge", location)

    def test_obsolete_converter_artifacts_are_removed(self):
        for name in ("config/prompts.py", "langflow/component_smoke.py", "example_config.json"):
            with self.subTest(name=name):
                self.assertFalse((ROOT / name).exists(), name)

    def test_runtime_has_no_legacy_protocol_or_imports(self):
        for directory in ("negotiation_game", "consequences_model", "langflow/custom_components", "config"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    self.assert_current_runtime(path.read_text(), str(path.relative_to(ROOT)))

    def test_exported_flows_have_no_converter_nodes_or_embedded_parsers(self):
        paths = sorted((ROOT / "langflow").glob("Negotiation_MVP*.json"))
        self.assertEqual(len(paths), 2)
        for path in paths:
            flow = json.loads(path.read_text())["data"]
            for node in flow["nodes"]:
                location = f"{path.name}:{node['id']}"
                self.assertNotIn("converter", node["data"]["type"].lower(), location)
                code = node["data"].get("node", {}).get("template", {}).get("code", {}).get("value")
                if code:
                    self.assert_current_runtime(code, location)
            for edge in flow["edges"]:
                self.assertNotEqual(edge.get("data", {}).get("targetHandle", {}).get("fieldName"),
                                    "converter_model", path.name)


if __name__ == "__main__":
    unittest.main()
