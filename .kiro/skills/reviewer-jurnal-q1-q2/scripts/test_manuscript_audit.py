#!/usr/bin/env python3

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manuscript_audit import audit, read_paragraphs  # noqa: E402


CLEAN_IJCCE = """Abstract
This study evaluates a controlled recognition protocol.
Keywords
palm vein; edge inference
1. Introduction
The engineering objective is stated with a bounded claim.
2. Methods
The split and training procedure were fixed before evaluation.
3. Results and Discussion
The measured result is reported with its evaluation condition.
4. Conclusions
The conclusion remains limited to the tested dataset and device.
CRediT authorship contribution statement
The authors report their actual roles.
Declaration of competing interest
The authors confirm the completed declaration separately.
Data availability
Access follows the dataset license.
References
Verified references appear here.
"""


class ManuscriptAuditTests(unittest.TestCase):
    def write(self, root: Path, name: str, text: str) -> Path:
        path = root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_clean_ijcce_has_no_major_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manuscript = self.write(root, "clean.md", CLEAN_IJCCE)
            report = audit(manuscript, target="ijcce", language="en", ai_use="none")
            self.assertEqual(report["counts"]["fatal"], 0)
            self.assertEqual(report["counts"]["major"], 0)

    def test_placeholders_promotional_language_and_mixed_dialect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text = CLEAN_IJCCE + "\n[insert result]\nThis groundbreaking model guarantees success.\nBehavior and behaviour were mixed."
            manuscript = self.write(root, "draft.md", text)
            report = audit(manuscript, target="ijcce", language="en", ai_use="none")
            codes = {item["code"] for item in report["findings"]}
            self.assertIn("unresolved-placeholder", codes)
            self.assertIn("claim-needs-scope", codes)
            self.assertIn("mixed-english-dialect", codes)

    def test_substantive_ai_requires_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            manuscript = self.write(Path(tmp), "draft.md", CLEAN_IJCCE)
            report = audit(manuscript, target="ijcce", ai_use="substantive")
            self.assertIn("missing-ai-declaration", {item["code"] for item in report["findings"]})

    def test_substantive_ai_declaration_must_precede_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = CLEAN_IJCCE + "\nDeclaration of generative AI and AI-assisted technologies\nReviewed by the authors."
            manuscript = self.write(Path(tmp), "misplaced.md", text)
            report = audit(manuscript, target="ijcce", ai_use="substantive")
            self.assertIn("misplaced-ai-declaration", {item["code"] for item in report["findings"]})

    def test_highlights_count_and_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manuscript = self.write(root, "clean.md", CLEAN_IJCCE)
            highlights = self.write(
                root,
                "highlights.txt",
                "Highlights\n- short item\n- " + ("x" * 86) + "\n",
            )
            report = audit(manuscript, target="ijcce", highlights=highlights)
            codes = {item["code"] for item in report["findings"]}
            self.assertIn("invalid-highlight-count", codes)
            self.assertIn("highlight-too-long", codes)

    def test_minimal_docx_text_extraction(self):
        xml = """<?xml version='1.0' encoding='UTF-8'?>
        <w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
          <w:body><w:p><w:r><w:t>Abstract</w:t></w:r></w:p></w:body>
        </w:document>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "minimal.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml)
            self.assertEqual(read_paragraphs(path), ["Abstract"])


if __name__ == "__main__":
    unittest.main()
