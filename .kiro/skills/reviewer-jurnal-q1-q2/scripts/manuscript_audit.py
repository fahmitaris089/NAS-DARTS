#!/usr/bin/env python3
"""Deterministic manuscript checks for author review, not AI/plagiarism scoring."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
SEVERITY_ORDER = {"fatal": 3, "major": 2, "minor": 1}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    paragraph: int | None = None
    excerpt: str | None = None


def read_paragraphs(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix != ".docx":
        raise ValueError(f"Unsupported input format: {suffix or '[none]'}")

    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(f"Invalid DOCX: {path}") from exc

    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{{{WORD_NS}}}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{WORD_NS}}}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{{{WORD_NS}}}tab":
                parts.append("\t")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def excerpt(text: str, limit: int = 140) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def add_matches(
    findings: list[Finding],
    paragraphs: Iterable[str],
    patterns: dict[str, str],
    severity: str,
    code: str,
    message_prefix: str,
) -> None:
    for index, paragraph in enumerate(paragraphs, start=1):
        for label, pattern in patterns.items():
            if re.search(pattern, paragraph, flags=re.IGNORECASE):
                findings.append(
                    Finding(
                        severity,
                        code,
                        f"{message_prefix}: {label}",
                        index,
                        excerpt(paragraph),
                    )
                )


def audit_placeholders(paragraphs: list[str]) -> list[Finding]:
    patterns = {
        "bracketed placeholder": r"\[[^\]]*(?:pending|insert|confirm|citation required|result|accuracy|latency|model size|standard deviation|co-author|email|department|butuh sitasi|masukkan|isi sesuai|jelaskan)[^\]]*\]",
        "TODO/TBD marker": r"\b(?:TODO|TBD|FIXME)\b",
    }
    findings: list[Finding] = []
    add_matches(findings, paragraphs, patterns, "major", "unresolved-placeholder", "Unresolved manuscript marker")
    return findings


def audit_promotional_claims(paragraphs: list[str]) -> list[Finding]:
    patterns = {
        "groundbreaking/revolutionary wording": r"\b(?:groundbreaking|revolutionary|unprecedented|revolusioner)\b",
        "absolute guarantee": r"\b(?:guarantees?\s+(?:success|superiority|acceptance)|always outperforms?|pasti|sempurna)\b",
        "unqualified SOTA claim": r"\bstate[- ]of[- ]the[- ]art\b",
        "unqualified best claim": r"\b(?:the best model|best-performing model|model terbaik)\b",
    }
    findings: list[Finding] = []
    add_matches(
        findings,
        paragraphs,
        patterns,
        "minor",
        "claim-needs-scope",
        "Verify evidence and scope for promotional or absolute wording",
    )
    return findings


def audit_formulaic_language(paragraphs: list[str]) -> list[Finding]:
    phrases = {
        "it is worth noting that": "It is worth noting that",
        "it is important to note": "It is important to note",
        "in today's rapidly evolving": "In today's rapidly evolving",
        "the results clearly demonstrate": "The results clearly demonstrate",
        "berdasarkan hasil tersebut": "Berdasarkan hasil tersebut",
        "dapat dilihat bahwa": "Dapat dilihat bahwa",
        "hal ini menunjukkan bahwa": "Hal ini menunjukkan bahwa",
    }
    joined = "\n".join(paragraphs).lower()
    findings: list[Finding] = []
    for needle, label in phrases.items():
        count = joined.count(needle)
        if count >= 2:
            findings.append(
                Finding(
                    "minor",
                    "formulaic-repetition",
                    f"Formulaic phrase appears {count} times; retain only where it adds meaning: {label}",
                )
            )
    return findings


def audit_repeated_openers(paragraphs: list[str]) -> list[Finding]:
    openings: list[str] = []
    for paragraph in paragraphs:
        words = re.findall(r"[A-Za-zÀ-ÿ]+", paragraph.lower())
        if len(words) >= 8:
            openings.append(" ".join(words[:3]))
    findings: list[Finding] = []
    for opening, count in Counter(openings).most_common():
        if count >= 3:
            findings.append(
                Finding(
                    "minor",
                    "repeated-paragraph-opener",
                    f"Paragraph opening '{opening}' occurs {count} times; check whether the rhetorical structure is repetitive.",
                )
            )
    return findings


def audit_english_dialect(text: str, language: str) -> list[Finding]:
    if language != "en":
        return []
    pairs = [
        ("behavior", "behaviour"),
        ("color", "colour"),
        ("analyze", "analyse"),
        ("modeling", "modelling"),
        ("optimization", "optimisation"),
        ("center", "centre"),
    ]
    findings: list[Finding] = []
    for american, british in pairs:
        has_us = re.search(rf"\b{american}\w*\b", text, flags=re.IGNORECASE)
        has_uk = re.search(rf"\b{british}\w*\b", text, flags=re.IGNORECASE)
        if has_us and has_uk:
            findings.append(
                Finding(
                    "minor",
                    "mixed-english-dialect",
                    f"Mixed English variants detected: {american}/{british}.",
                )
            )
    return findings


def normalized_heading(paragraph: str) -> str:
    value = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", paragraph).strip().lower()
    return re.sub(r"[^a-z ]", "", value)


def has_heading(headings: list[str], alternatives: tuple[str, ...]) -> bool:
    return any(any(heading == alt or heading.startswith(alt + " ") for alt in alternatives) for heading in headings)


def audit_ijcce_sections(paragraphs: list[str]) -> list[Finding]:
    headings = [normalized_heading(p) for p in paragraphs if len(p.split()) <= 12]
    required = {
        "abstract": ("abstract",),
        "keywords": ("keywords", "index terms"),
        "introduction": ("introduction",),
        "methods": ("material and methods", "materials and methods", "methods", "methodology", "proposed methodology"),
        "results": ("results", "results and discussion"),
        "discussion": ("discussion", "results and discussion"),
        "conclusions": ("conclusion", "conclusions"),
        "references": ("references",),
    }
    findings: list[Finding] = []
    for label, alternatives in required.items():
        if not has_heading(headings, alternatives):
            findings.append(Finding("major", "missing-core-section", f"IJCCE core section not detected: {label}."))

    end_matter = {
        "CRediT/author contributions": ("credit authorship contribution statement", "author contributions", "credit statement"),
        "competing interest": ("declaration of competing interest", "conflict of interest", "competing interests"),
        "data availability": ("data availability", "data and code availability", "availability of data and materials"),
    }
    for label, alternatives in end_matter.items():
        if not has_heading(headings, alternatives):
            findings.append(Finding("minor", "missing-end-matter", f"Submission end matter not detected: {label}."))
    return findings


def audit_ai_declaration(paragraphs: list[str], ai_use: str) -> list[Finding]:
    text = "\n".join(paragraphs).lower()
    declaration_present = "declaration of generative ai" in text or "declaration of ai-assisted" in text
    tool_terms = re.search(r"\b(?:chatgpt|codex|gemini|claude|large language model|generative ai|ai-assisted tool)\b", text)

    if ai_use == "substantive" and not declaration_present:
        return [
            Finding(
                "major",
                "missing-ai-declaration",
                "Substantive AI use was selected, but no generative-AI manuscript declaration was detected before references.",
            )
        ]
    if ai_use == "substantive" and declaration_present:
        declaration_index = next(
            index
            for index, paragraph in enumerate(paragraphs)
            if "declaration of generative ai" in paragraph.lower()
            or "declaration of ai-assisted" in paragraph.lower()
        )
        reference_indices = [
            index for index, paragraph in enumerate(paragraphs) if normalized_heading(paragraph) == "references"
        ]
        if reference_indices and declaration_index > reference_indices[0]:
            return [
                Finding(
                    "major",
                    "misplaced-ai-declaration",
                    "The AI declaration was detected after References; place it immediately before the reference list.",
                )
            ]
    if ai_use == "research-method" and not tool_terms:
        return [
            Finding(
                "major",
                "missing-ai-method-reporting",
                "AI use in the research method was selected, but reproducible tool reporting was not detected.",
            )
        ]
    if ai_use == "none" and declaration_present:
        return [
            Finding(
                "minor",
                "ai-use-setting-conflict",
                "Audit setting says no AI use, but an AI declaration is present; confirm the actual workflow.",
            )
        ]
    return []


def clean_highlight(text: str) -> str:
    return re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", text).strip()


def audit_highlights(path: Path) -> tuple[list[Finding], dict[str, object]]:
    paragraphs = read_paragraphs(path)
    items = [clean_highlight(p) for p in paragraphs if clean_highlight(p).lower() != "highlights"]
    findings: list[Finding] = []
    if not 3 <= len(items) <= 5:
        findings.append(
            Finding(
                "major",
                "invalid-highlight-count",
                f"IJCCE highlights require 3–5 bullets; detected {len(items)}.",
            )
        )
    for index, item in enumerate(items, start=1):
        if len(item) > 85:
            findings.append(
                Finding(
                    "major",
                    "highlight-too-long",
                    f"Highlight {index} has {len(item)} characters; maximum is 85 including spaces.",
                    index,
                    excerpt(item),
                )
            )
    return findings, {"count": len(items), "character_counts": [len(item) for item in items]}


def audit(
    manuscript: Path,
    target: str = "generic",
    language: str = "en",
    ai_use: str = "none",
    highlights: Path | None = None,
) -> dict[str, object]:
    paragraphs = read_paragraphs(manuscript)
    text = "\n".join(paragraphs)
    findings: list[Finding] = []
    findings.extend(audit_placeholders(paragraphs))
    findings.extend(audit_promotional_claims(paragraphs))
    findings.extend(audit_formulaic_language(paragraphs))
    findings.extend(audit_repeated_openers(paragraphs))
    findings.extend(audit_english_dialect(text, language))
    findings.extend(audit_ai_declaration(paragraphs, ai_use))
    if target == "ijcce":
        findings.extend(audit_ijcce_sections(paragraphs))

    highlight_metadata: dict[str, object] | None = None
    if highlights is not None:
        highlight_findings, highlight_metadata = audit_highlights(highlights)
        findings.extend(highlight_findings)

    findings.sort(key=lambda item: (-SEVERITY_ORDER[item.severity], item.code, item.paragraph or 0))
    counts = Counter(item.severity for item in findings)
    return {
        "manuscript": str(manuscript),
        "target": target,
        "language": language,
        "ai_use": ai_use,
        "metadata": {
            "paragraphs": len(paragraphs),
            "words": len(re.findall(r"\b\w+\b", text)),
            "highlights": highlight_metadata,
        },
        "counts": {severity: counts.get(severity, 0) for severity in ("fatal", "major", "minor")},
        "findings": [asdict(item) for item in findings],
        "limitations": [
            "This audit does not calculate plagiarism, similarity, authorship, or AI-detector scores.",
            "Citation support, novelty, methodological validity, and journal fit require human review.",
        ],
    }


def to_markdown(report: dict[str, object]) -> str:
    counts = report["counts"]
    metadata = report["metadata"]
    lines = [
        "# Manuscript Audit",
        "",
        f"- File: `{report['manuscript']}`",
        f"- Target: `{report['target']}`",
        f"- Language: `{report['language']}`",
        f"- AI-use setting: `{report['ai_use']}`",
        f"- Paragraphs/words: {metadata['paragraphs']}/{metadata['words']}",
        f"- Findings: fatal={counts['fatal']}, major={counts['major']}, minor={counts['minor']}",
        "",
    ]
    findings = report["findings"]
    if not findings:
        lines.extend(["No deterministic findings.", ""])
    else:
        for severity in ("fatal", "major", "minor"):
            selected = [item for item in findings if item["severity"] == severity]
            if not selected:
                continue
            lines.extend([f"## {severity.title()}", ""])
            for item in selected:
                location = f" (paragraph {item['paragraph']})" if item["paragraph"] else ""
                lines.append(f"- **{item['code']}**{location}: {item['message']}")
                if item["excerpt"]:
                    lines.append(f"  - Excerpt: `{item['excerpt']}`")
            lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--target", choices=("generic", "ijcce"), default="generic")
    parser.add_argument("--language", choices=("en", "id"), default="en")
    parser.add_argument(
        "--ai-use",
        choices=("none", "grammar-only", "substantive", "research-method"),
        default="none",
    )
    parser.add_argument("--highlights", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.manuscript, args.highlights):
        if path is not None and not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
    try:
        report = audit(args.manuscript, args.target, args.language, args.ai_use, args.highlights)
    except (OSError, UnicodeError, ValueError, ET.ParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(to_markdown(report))
    counts = report["counts"]
    return 1 if counts["fatal"] or counts["major"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
