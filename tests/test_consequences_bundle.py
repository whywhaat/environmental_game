"""Получатель архива может запустить модель без репозитория и окружения разработки."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from consequences_model.build_bundle import build_bundle


class ConsequencesBundleTests(unittest.TestCase):
    def test_extracted_package_runs_api_and_cli_without_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = build_bundle(root / "model.zip")
            extracted = root / "receiver"
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
            process = subprocess.run(
                [sys.executable, "-E", "-S", "-m", "consequences_model",
                 "examples/consequences/open_river.json", "--json"],
                cwd=extracted, capture_output=True, text=True, check=True)
            result = json.loads(process.stdout)
            self.assertEqual(result["outcome"]["id"], "open_river")
            self.assertEqual(result["paid"], 270_000_000)
            self.assertEqual(result["commissioning_month"], 36)
            program = '''
import json
from pathlib import Path
from consequences_model import forecast
results = {}
for case in json.loads(Path("examples/consequences/manifest.json").read_text(encoding="utf-8")):
    data = json.loads(Path("examples/consequences", case["id"] + ".json").read_text(encoding="utf-8"))
    result = forecast(data)
    results[case["id"]] = {"id": result["outcome"]["id"], "paid": result["paid"],
                           "accepted": result["protocol"]["accepted"], "month": result["commissioning_month"]}
print(json.dumps(results))
'''
            process = subprocess.run([sys.executable, "-E", "-S", "-c", program], cwd=extracted,
                                     capture_output=True, text=True, check=True)
            results = json.loads(process.stdout)
            self.assertEqual(len(results), 10)
            for expected_id, actual in results.items():
                self.assertEqual(actual["id"], expected_id)
            self.assertFalse(results["each_for_self"]["accepted"])
            self.assertEqual(results["each_for_self"]["paid"], 90_000_000)
            self.assertIsNone(results["each_for_self"]["month"])
            for case, accepted, outcome, paid, month in (
                    ("accepted", True, "compromise", 260_000_000, 60),
                    ("rejected", False, "each_for_self", 115_000_000, None)):
                with self.subTest(real_game=case):
                    process = subprocess.run(
                        [sys.executable, "-E", "-S", "-m", "consequences_model",
                         f"examples/integration/vlad_10a5c68_{case}.json", "--json"],
                        cwd=extracted, capture_output=True, text=True)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    result = json.loads(process.stdout)
                    self.assertEqual(result["protocol"]["accepted"], accepted)
                    self.assertEqual(result["outcome"]["id"], outcome)
                    self.assertEqual(result["paid"], paid)
                    self.assertEqual(result["commissioning_month"], month)
                    self.assertEqual(result["warnings"], [])
                    self.assertEqual(len(result["herald_text"].splitlines()), 1)
            partial = extracted / "failed-game.json"
            partial.write_text(json.dumps({"status": "failed", "state": None}))
            process = subprocess.run(
                [sys.executable, "-E", "-S", "-m", "consequences_model", partial.name, "--json"],
                cwd=extracted, capture_output=True, text=True, check=True)
            skipped = json.loads(process.stdout)
            self.assertEqual(skipped["status"], "skipped")
            self.assertNotIn("outcome", skipped)
            self.assertIn("не завершена", skipped["herald_text"])

    def test_archive_contains_transfer_materials_and_integrity_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = build_bundle(Path(directory) / "model.zip")
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
                for name in ("README.md", "docs/consequences-integration.md", "examples/integration/README.md",
                             "langflow/Consequences_demo.json",
                             "langflow/custom_components/my_components/ConsequencesModel.py"):
                    self.assertIn(name, names)
                for case in json.loads(bundle.read("examples/consequences/manifest.json")):
                    self.assertIn(f"assets/consequences/{case['id']}.png", names)
                for name in names:
                    self.assertFalse(set(Path(name).parts) & {"games", ".git", ".env", "__pycache__", "negotiation_mvp"})
                    self.assertNotEqual(Path(name).name, "dialog.py")
                    self.assertNotEqual(name, "langflow/Negotiation_MVP_5_players.json")
                manifest = json.loads(bundle.read("manifest.json"))
                self.assertEqual(set(manifest["files"]), names - {"manifest.json"})
                for name, digest in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), digest, name)
                flow = json.loads(bundle.read("langflow/Consequences_demo.json"))
                model = next(n for n in flow["data"]["nodes"] if n["data"]["type"] == "ConsequencesModel")
                self.assertEqual(model["data"]["node"]["template"]["code"]["value"],
                                 bundle.read("langflow/custom_components/my_components/ConsequencesModel.py").decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
