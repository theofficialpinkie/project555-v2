#!/usr/bin/env python3
"""
Construction-plan PDF extractor and concrete-callout highlighter.

Extracts, per PDF page:
- sheet_title: title-block title in the bottom-right of the sheet
- drawing_titles: large and/or underlined view titles inside the sheet
- callouts: text labels likely associated with leader lines or detail pointers

Also supports highlighting keyword mentions only when they occur inside detected
callouts (the MVP defaults to the keyword "concrete").

CLI examples:
  python pdf_plan_extractor.py input.pdf -o extracted_plan_data.json --csv out.csv
  python pdf_plan_extractor.py input.pdf --highlight-keyword concrete --highlight-dir highlighted

Requires:
  pip install pymupdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import fitz  # PyMuPDF

STOP_TITLE = {
    "NOTES",
    "NOTE",
    "LEGEND",
    "DESCRIPTION OF ALTERATIONS",
    "AS-BUILT REVISIONS",
    "CONTRACT NUMBER",
    "DRAWING NO",
    "SHEET NO",
    "REGION",
    "COUNTY",
    "PIN",
    "BRIDGES",
    "CULVERTS",
    "ALL DIMENSIONS IN FT UNLESS OTHERWISE NOTED",
    "ALL DIMENSIONS IN ft UNLESS OTHERWISE NOTED",
    "JOB MANAGER",
    "DESIGN SUPERVISOR",
}

CALLOUT_NOISE = re.compile(
    r"^(\d+[\-+]?\d*|\d+\s*0\s*\d+|\d+\"|\d+'|[A-Z]$|[A-Z]-[A-Z]$|NTS|TYP\.?|\(?TYP\)?|MAX|MIN)$",
    re.I,
)

# Strong title words describe a drawing/view, rather than a note block or table heading.
DRAWING_TITLE_WORDS = re.compile(
    r"\b(PLAN|ELEVATION|SECTION|DETAIL|VIEW|PROFILE|LAYOUT|DIAGRAM|SCHEDULE)\b",
    re.I,
)

TITLE_PREFIX_NOISE = re.compile(
    r"^(SEE|END DIMENSION|FOR DETAILS|FOR |NOTE\b|NOTES\b|ITEM\b|EL\.?\b|ELEV\.?\b|STA\.?\b|AZ\b|TYP\b|APPROX\.?\b|PROPOSED\b)",
    re.I,
)

METADATA_WORDS = (
    "BOULEVARD",
    "STREET",
    "STATION LINE",
    "DESCRIPTION",
    "UNIT",
    "QUANTITY",
    "PLACEMENT",
    "ESTIMATED",
    "FINAL",
    "CONTRACT NUMBER",
    "DRAWING NO",
    "SHEET NO",
    "ALL DIMENSIONS",
)


# Concrete phrases that Carol identified as belonging to non-Item-555 work.
# These rules apply only to a CONCRETE match; the other configured keywords
# (grout, joint filler, wire fabric, etc.) remain independently reviewable.
CONCRETE_NON_555_EXCLUSIONS = [
    "Removal",
    "Fill",
    "Paved areas",
    "Pavement",
    "Paving",
    "Asphalt",
    "Disposal of",
    "Cold milling",
    "Cold mill",
    "Portland",
    "PCC",
    "Precast",
    "Sawing",
    "Saw cut",
    "Saw cutting",
    "Piles",
    "Pile",
    "Shafts",
    "Soldier pile and lagging",
    "Reinforcing steel",
    "Bar reinforcement",
    "Approach",
    "Slab",
    "Deck",
    "Reinforced",
    "Polymer",
    "Sidewalks",
    "Sidewalk",
    "Curb",
    "Barrier",
    "Fence",
    "Prefabricated",
    "Lightweight",
    "Ultra-high performance",
    "UHPC",
    "Overlay",
    "Joint header",
    "Hybrid Composite Synthetic",
    "HCSC",
    "Painting",
    "Sealing",
    "Protective sealing",
    "Coating",
    "Sealer",
    "Masonry",
    "Apron",
    "Prestressed",
    "Post-tensioned",
    "Elastomeric",
    "Railing",
    "Parapet",
    "Replacement of",
    "Friction surface",
    "Pavers",
    "Gutters",
    "Impact Attenuator",
    "Abandon",
    "Pedestrian signal pole",
    "Pullbox",
]

# An explicit Item 555 reference is stronger evidence than a nearby exclusion
# term and therefore keeps the concrete mention in the flagged set.
ITEM_555_RE = re.compile(
    r"(?:\bITEM(?:\s+NO\.?)?\s*555(?:\.\d+)?\b|\b555\.\d+\b)",
    re.I,
)


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s.replace("ﬁ", "fi").replace("ﬂ", "fl")


def _normalized_word(value: str) -> str:
    """Normalize a PDF/OCR token for conservative fuzzy keyword matching."""
    return re.sub(r"[^A-Z0-9]", "", clean_text(value).upper())


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Return True when two short strings differ by no more than one edit."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False

    # Same-length case: one substituted / badly encoded glyph.
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1

    # One insertion / deletion.
    short, long = (left, right) if len(left) < len(right) else (right, left)
    i = j = edits = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1
    return True


def _token_match_quality(token: str, keyword: str) -> str | None:
    """Return how a source token matches a configured keyword.

    In addition to exact and conservative PDF/OCR recovery, ``CONC.`` /
    ``conc.`` are recognized as whole-word abbreviations for ``Concrete``.
    The punctuation is stripped by normalization, so both forms resolve to
    the token ``CONC``.
    """
    token_norm = _normalized_word(token)
    keyword_norm = _normalized_word(keyword)
    if not token_norm or not keyword_norm:
        return None
    if token_norm == keyword_norm:
        return "exact"

    # Engineering abbreviation. Keep it canonicalized under the Concrete
    # keyword so Carol's non-Item-555 exclusions still apply.
    if keyword_norm == "CONCRETE" and token_norm == "CONC":
        return "recognized abbreviation CONC."

    # Short construction terms such as GROUT can lose their final rendered
    # glyph in the PDF text layer. Do not use general edit distance here: a
    # strict trailing-glyph rule keeps accidental matches low.
    if 5 <= len(keyword_norm) <= 6:
        if (
            keyword_norm[-1] != "S"
            and len(token_norm) == len(keyword_norm) - 1
            and keyword_norm.startswith(token_norm)
        ):
            return "recovered one-character text error"
        return None

    if len(keyword_norm) < 7 or len(token_norm) < len(keyword_norm) - 1:
        return None
    if _edit_distance_at_most_one(token_norm, keyword_norm):
        return "recovered one-character text error"
    return None


def _token_matches_keyword(token: str, keyword: str) -> bool:
    """Boolean wrapper around :func:`_token_match_quality`."""
    return _token_match_quality(token, keyword) is not None


def _keyword_tokens(value: str) -> list[str]:
    return [
        _normalized_word(token)
        for token in re.findall(r"[A-Za-z0-9]+", value or "")
        if _normalized_word(token)
    ]


def normalize_keywords(keywords: Iterable[str] | str) -> list[str]:
    """Clean and deduplicate user-supplied keywords while preserving order."""
    if isinstance(keywords, str):
        raw = re.split(r"[\n,]+", keywords)
    else:
        raw = list(keywords)
    result: list[str] = []
    seen: set[str] = set()
    for value in raw:
        cleaned = clean_text(str(value)).strip(" ,;")
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _text_contains_keyword(text: str, keyword: str) -> bool:
    text_tokens = _keyword_tokens(text)
    keyword_tokens = _keyword_tokens(keyword)
    if not keyword_tokens or len(text_tokens) < len(keyword_tokens):
        return False
    width = len(keyword_tokens)
    for index in range(len(text_tokens) - width + 1):
        if all(
            _token_matches_keyword(text_tokens[index + offset], target)
            for offset, target in enumerate(keyword_tokens)
        ):
            return True
    return False




def _concrete_exclusion_terms(text: str) -> list[str]:
    """Return non-555 terms found in the local concrete callout context.

    Matching is whole-word / whole-phrase and case-insensitive.  A narrow
    prefix rule is used for ``abandon`` so it also catches ``abandoned``.
    """
    if ITEM_555_RE.search(text or ""):
        return []

    matches: list[str] = []
    normalized = clean_text(text)
    for phrase in CONCRETE_NON_555_EXCLUSIONS:
        if phrase.casefold() == "abandon":
            if re.search(r"\babandon(?:ed|ing|ment)?\b", normalized, re.I):
                matches.append(phrase)
            continue
        if _text_contains_keyword(normalized, phrase):
            matches.append(phrase)
    return matches


def _keyword_is_concrete(keyword: str) -> bool:
    return _normalized_word(keyword) == "CONCRETE"

def _repair_keyword_text(text: str, keyword: str) -> str:
    """Repair one-character PDF/OCR errors inside a matched word or phrase."""
    pieces = re.split(r"([A-Za-z0-9]+)", text or "")
    token_positions = [index for index in range(1, len(pieces), 2)]
    keyword_tokens = re.findall(r"[A-Za-z0-9]+", keyword or "")
    if not keyword_tokens:
        return clean_text(text)

    for start_index in range(len(token_positions) - len(keyword_tokens) + 1):
        positions = token_positions[start_index : start_index + len(keyword_tokens)]
        source_tokens = [pieces[position] for position in positions]
        if not all(
            _token_matches_keyword(source, target)
            for source, target in zip(source_tokens, keyword_tokens)
        ):
            continue
        for position, source, target in zip(positions, source_tokens, keyword_tokens):
            # CONC. is a valid abbreviation, not a damaged PDF token. Preserve
            # the source wording in the report while mapping it to Concrete.
            if _token_match_quality(source, target) == "recognized abbreviation CONC.":
                continue
            if source.isupper():
                pieces[position] = target.upper()
            elif source[:1].isupper():
                pieces[position] = target.capitalize()
            else:
                pieces[position] = target.lower()
        break
    return clean_text("".join(pieces)).replace(" ,", ",")


def _analysis_rect(page: fitz.Page) -> fitz.Rect:
    """Return PyMuPDF's unrotated page coordinate space.

    Text, vector drawings, and annotations use unrotated coordinates even when
    ``page.rect`` reflects a 90/270-degree display rotation. Using ``cropbox``
    keeps all extraction geometry in the same coordinate system.
    """
    return fitz.Rect(page.cropbox)


def _prepare_page_text(
    page: fitz.Page,
    *,
    ocr_if_sparse: bool = True,
    ocr_dpi: int = 200,
) -> tuple[fitz.TextPage | None, dict[str, Any]]:
    """Use native PDF text, with full-page OCR fallback for image-only pages.

    Many contract-plan PDFs are mixed: some sheets contain searchable CAD text,
    while others are flattened page images with only a sheet number/date left as
    native text. Those pages previously appeared in the report but had empty
    analysis.
    """
    native_text = page.get_text("text")
    native_words = page.get_text("words")
    alpha_chars = sum(char.isalpha() for char in native_text)
    sparse = len(native_words) < 20 or alpha_chars < 80

    info: dict[str, Any] = {
        "text_source": "native",
        "ocr_used": False,
        "native_word_count": len(native_words),
        "extracted_word_count": len(native_words),
        "analysis_status": "analyzed",
        "ocr_error": None,
    }
    if not (ocr_if_sparse and sparse):
        return None, info

    try:
        textpage = page.get_textpage_ocr(
            language="eng",
            dpi=ocr_dpi,
            full=True,
        )
        ocr_words = page.get_text("words", textpage=textpage)
        info.update(
            {
                "text_source": "ocr",
                "ocr_used": True,
                "extracted_word_count": len(ocr_words),
                "analysis_status": "analyzed with OCR",
            }
        )
        return textpage, info
    except Exception as exc:  # keep native extraction alive if OCR is unavailable
        info.update(
            {
                "text_source": "native_sparse",
                "analysis_status": "limited: OCR unavailable",
                "ocr_error": clean_text(str(exc)),
            }
        )
        return None, info


def line_text_from_page(
    page: fitz.Page, textpage: fitz.TextPage | None = None
) -> list[dict[str, Any]]:
    """Return joined text lines with bbox and median font size."""
    data = page.get_text("dict", textpage=textpage) if textpage else page.get_text("dict")
    lines: list[dict[str, Any]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if clean_text(span.get("text"))]
            if not spans:
                continue
            spans = sorted(spans, key=lambda span: span["bbox"][0])
            text = clean_text(" ".join(clean_text(span["text"]) for span in spans))
            if not text:
                continue
            x0 = min(span["bbox"][0] for span in spans)
            y0 = min(span["bbox"][1] for span in spans)
            x1 = max(span["bbox"][2] for span in spans)
            y1 = max(span["bbox"][3] for span in spans)
            size = median([span.get("size", 0) for span in spans])
            source_rect = fitz.Rect(x0, y0, x1, y1)
            display_rect = source_rect * page.rotation_matrix if page.rotation else source_rect
            lines.append(
                {
                    "text": text,
                    "bbox": [x0, y0, x1, y1],
                    "display_bbox": [
                        display_rect.x0,
                        display_rect.y0,
                        display_rect.x1,
                        display_rect.y1,
                    ],
                    "font_size": round(float(size), 2),
                }
            )
    return lines


def norm_box(box: list[float], width: float, height: float) -> list[float]:
    return [
        round(box[0] / width, 4),
        round(box[1] / height, 4),
        round(box[2] / width, 4),
        round(box[3] / height, 4),
    ]


def _title_line_box(line: dict[str, Any]) -> list[float]:
    return line.get("display_bbox", line["bbox"])


def _compact_label(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _find_title_block_anchor(
    lines: list[dict[str, Any]],
    width: float,
    height: float,
    label: str,
) -> dict[str, Any] | None:
    target = _compact_label(label)
    matches: list[dict[str, Any]] = []
    for line in lines:
        x0, y0, x1, y1 = _title_line_box(line)
        if y0 < 0.82 * height or x0 < 0.55 * width:
            continue
        compact = _compact_label(line.get("text", ""))
        if target in compact:
            matches.append(line)
    if not matches:
        return None
    # Prefer the rightmost match in the actual title block. This avoids an
    # occasional note/table reference elsewhere along the bottom of the page.
    return max(matches, key=lambda item: (_title_line_box(item)[0], _title_line_box(item)[1]))


def _merge_title_baselines(
    candidates: list[dict[str, Any]],
    *,
    height: float,
) -> list[dict[str, Any]]:
    """Deduplicate and merge title fragments that share the same baseline."""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for line in candidates:
        box = _title_line_box(line)
        key = (
            clean_text(line.get("text", "")).upper(),
            round(box[0]),
            round(box[1]),
            round(box[2]),
            round(box[3]),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)

    unique.sort(key=lambda item: (_title_line_box(item)[1], _title_line_box(item)[0]))
    rows: list[list[dict[str, Any]]] = []
    baseline_tolerance = max(2.5, 0.0045 * height)
    for line in unique:
        cy = (_title_line_box(line)[1] + _title_line_box(line)[3]) / 2
        if rows:
            row_cy = sum(
                (_title_line_box(item)[1] + _title_line_box(item)[3]) / 2
                for item in rows[-1]
            ) / len(rows[-1])
            if abs(cy - row_cy) <= baseline_tolerance:
                rows[-1].append(line)
                continue
        rows.append([line])

    merged: list[dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda item: _title_line_box(item)[0])
        text = clean_text(" ".join(item["text"] for item in row))
        boxes = [_title_line_box(item) for item in row]
        merged.append(
            {
                "text": text,
                "display_bbox": [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
                "font_size": max(float(item.get("font_size", 0)) for item in row),
            }
        )
    return merged


def extract_sheet_title_info(
    lines: list[dict[str, Any]], width: float, height: float
) -> dict[str, Any]:
    """Extract every line inside the lower-right sheet-title cell.

    The reliable signal is the title-block geometry, not title wording or font
    size. NYSDOT title cells can contain one to four stacked lines, for example:

      CONC. DECK REPAIR TYPE CDR-3
      BOTTOM OF DECK
      OVERHEAD PATCHING/SPALL REPAIR

    or:

      GENERAL PLAN
      LOCATION 2 - NY25A (NORTHERN BLVD)
      AT COMMUNITY DRIVE

    This routine anchors the cell using ALL DIMENSIONS, DRAWING NO., SHEET NO.,
    and CONTRACT NUMBER labels. A normalized fallback handles imperfect OCR.
    """
    all_dims = _find_title_block_anchor(lines, width, height, "ALL DIMENSIONS")
    drawing_no = _find_title_block_anchor(lines, width, height, "DRAWING NO")
    sheet_no = _find_title_block_anchor(lines, width, height, "SHEET NO")
    contract_no = _find_title_block_anchor(lines, width, height, "CONTRACT NUMBER")

    anchor_count = sum(anchor is not None for anchor in (all_dims, drawing_no, sheet_no, contract_no))
    if drawing_no:
        drawing_box = _title_line_box(drawing_no)
        right = drawing_box[0] - max(2.0, 0.002 * width)
    else:
        right = 0.91 * width

    if all_dims:
        all_dims_box = _title_line_box(all_dims)
        left = all_dims_box[0] - max(3.0, 0.003 * width)
        top = all_dims_box[3] + max(1.0, 0.0015 * height)
    else:
        # Across the sample title blocks, the title cell occupies the band just
        # left of DRAWING NO. and begins at roughly 78% of sheet width.
        left = max(0.745 * width, right - 0.145 * width)
        if contract_no:
            top = _title_line_box(contract_no)[3] + max(1.0, 0.0015 * height)
        else:
            top = 0.884 * height

    if sheet_no:
        bottom = _title_line_box(sheet_no)[3] + max(4.0, 0.006 * height)
    else:
        bottom = 0.947 * height

    # Guard against malformed OCR anchors.
    left = max(0.70 * width, min(left, 0.82 * width))
    right = max(left + 0.07 * width, min(right, 0.925 * width))
    top = max(0.865 * height, min(top, 0.91 * height))
    bottom = max(top + 0.025 * height, min(bottom, 0.955 * height))

    candidates: list[dict[str, Any]] = []
    for line in lines:
        box = _title_line_box(line)
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        text = clean_text(line.get("text", ""))
        upper = text.upper().strip(" :.")
        compact = _compact_label(text)

        if not (left <= cx <= right and top <= cy <= bottom):
            continue
        if not text or len(text) < 2:
            continue
        if any(
            label in compact
            for label in (
                "ALLDIMENSIONS",
                "DRAWINGNO",
                "SHEETNO",
                "CONTRACTNUMBER",
            )
        ):
            continue
        if upper in STOP_TITLE:
            continue
        if text.lower().endswith(".dgn"):
            continue
        if re.fullmatch(r"[0-9]+", text):
            continue
        if re.fullmatch(r"D[0-9]{6}", upper):
            continue
        # Sheet/drawing codes belong in the adjacent cell, not the title cell.
        if re.fullmatch(r"[A-Z]{1,5}[0-9]*-[A-Z0-9-]+", upper) and cx > right - 0.02 * width:
            continue
        # Small title lines are legitimate; 4 pt is common in title blocks.
        if float(line.get("font_size", 0)) < 3.8:
            continue
        candidates.append(line)

    merged = _merge_title_baselines(candidates, height=height)
    if not merged:
        return {
            "title": None,
            "lines": [],
            "bbox": None,
            "source": "title_block_anchors" if anchor_count >= 2 else "normalized_fallback",
            "confidence": "low",
        }

    # Keep vertical order and remove exact repeated text caused by duplicated
    # CAD text layers. Do not reset at larger gaps: the complete sheet title can
    # legitimately contain a large first-to-second-line gap.
    title_lines: list[str] = []
    selected: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for line in merged:
        text = clean_text(line["text"])
        key = text.upper()
        if key in seen_text:
            continue
        seen_text.add(key)
        title_lines.append(text)
        selected.append(line)

    boxes = [line["display_bbox"] for line in selected]
    bbox = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    confidence = "high" if anchor_count >= 3 else "medium" if anchor_count >= 1 else "low"
    return {
        "title": " / ".join(title_lines),
        "lines": title_lines,
        "bbox": bbox,
        "source": "title_block_anchors" if anchor_count >= 2 else "normalized_fallback",
        "confidence": confidence,
    }


def extract_sheet_title(lines: list[dict[str, Any]], width: float, height: float) -> str | None:
    """Backward-compatible string-only sheet-title helper."""
    return extract_sheet_title_info(lines, width, height)["title"]


def collect_line_segments(page: fitz.Page) -> list[tuple[float, float, float, float, float]]:
    segments: list[tuple[float, float, float, float, float]] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            if item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            length = math.hypot(p1.x - p2.x, p1.y - p2.y)
            if 8 <= length <= 300:
                segments.append((p1.x, p1.y, p2.x, p2.y, length))
    return segments


def _line_is_underlined(
    line: dict[str, Any], segments: list[tuple[float, float, float, float, float]]
) -> bool:
    x0, _, x1, y1 = line["bbox"]
    text_width = max(1.0, x1 - x0)
    for xa, ya, xb, yb, _ in segments:
        if abs(ya - yb) > 1.5:
            continue
        y = (ya + yb) / 2
        if not (y1 - 1 <= y <= y1 + 7):
            continue
        sx0, sx1 = sorted((xa, xb))
        overlap = max(0.0, min(x1, sx1) - max(x0, sx0))
        if overlap >= 0.45 * text_width:
            return True
    return False


def _uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(char.isupper() for char in letters) / len(letters)


def _join_stacked_titles(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join a short second title line such as SECTION A-A to the title directly above it."""
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.get("display_bbox", item["bbox"])[1],
            item.get("display_bbox", item["bbox"])[0],
        ),
    )
    consumed: set[int] = set()
    output: list[dict[str, Any]] = []

    for i, first in enumerate(ordered):
        if i in consumed:
            continue
        fx0, fy0, fx1, fy1 = first.get("display_bbox", first["bbox"])
        fcenter = (fx0 + fx1) / 2
        merged = dict(first)

        for j in range(i + 1, min(i + 5, len(ordered))):
            if j in consumed:
                continue
            second = ordered[j]
            sx0, sy0, sx1, sy1 = second.get("display_bbox", second["bbox"])
            gap = sy0 - fy1
            if gap < -1 or gap > 7:
                continue
            scenter = (sx0 + sx1) / 2
            centered = abs(fcenter - scenter) <= max(24, 0.12 * (fx1 - fx0))
            short_suffix = bool(
                re.match(
                    r"^(SECTION\s+[A-Z0-9]+(?:-[A-Z0-9]+)?|PLAN|ELEVATION|PROFILE|DETAIL|VIEW)$",
                    second["text"].upper(),
                )
            )
            if centered and short_suffix and len(first["text"]) > len(second["text"]):
                merged["text"] = f"{first['text']} - {second['text']}"
                source_first = first["bbox"]
                source_second = second["bbox"]
                merged["bbox"] = [
                    min(source_first[0], source_second[0]),
                    min(source_first[1], source_second[1]),
                    max(source_first[2], source_second[2]),
                    max(source_first[3], source_second[3]),
                ]
                merged["display_bbox"] = [min(fx0, sx0), fy0, max(fx1, sx1), sy1]
                merged["font_size"] = round(max(first["font_size"], second["font_size"]), 2)
                consumed.add(j)
                break

        output.append(merged)
    return output


def extract_drawing_titles(
    lines: list[dict[str, Any]],
    segments: list[tuple[float, float, float, float, float]],
    width: float,
    height: float,
) -> list[dict[str, Any]]:
    """
    Extract drawing/view titles.

    Key change from the first version: a title-word match alone is no longer enough.
    Candidates must be visibly larger than body/callout text or genuinely underlined.
    This prevents note sentences such as "END DIMENSION IN ELEVATION..." from being
    misclassified while retaining low-page titles just above the title block.
    """
    body_sizes = [
        line["font_size"]
        for line in lines
        if 0.05 * width < line.get("display_bbox", line["bbox"])[0] < 0.96 * width
        and line.get("display_bbox", line["bbox"])[1] < 0.88 * height
    ]
    body_median = median(body_sizes) if body_sizes else 5.0
    large_threshold = max(6.3, body_median + 1.15)

    candidates: list[dict[str, Any]] = []
    for line in lines:
        text = line["text"].strip()
        upper = text.upper().strip(" :.")
        x0, y0, x1, _ = line.get("display_bbox", line["bbox"])

        # Include titles near the bottom of the drawing area, but stop before title block.
        if y0 > 0.88 * height or x0 < 0.04 * width or x1 > 0.97 * width:
            continue
        if len(text) < 4 or upper in STOP_TITLE:
            continue
        if _uppercase_ratio(text) < 0.68:
            continue
        if TITLE_PREFIX_NOISE.match(upper):
            continue
        if any(word in upper for word in METADATA_WORDS):
            continue
        if re.match(r"^(#|S\s*\d+\s*\d|S\d+|\d|[A-Z]\.)", upper):
            continue
        if re.match(r"^([A-Z]\s+){2,}[A-Z]", text):
            continue

        underlined = _line_is_underlined(line, segments)
        has_title_word = bool(DRAWING_TITLE_WORDS.search(text))
        visibly_large = line["font_size"] >= large_threshold

        # Large all-caps labels with view/title words are the normal case.
        # Underlined labels can pass at a slightly lower size (e.g. WINGWALL 1).
        large_title_shape = visibly_large and (has_title_word or (len(text) >= 9 and len(text.split()) >= 2))
        if not (large_title_shape or (underlined and line["font_size"] >= body_median + 0.35)):
            continue

        # Sentences and note fragments are unlikely to be drawing titles.
        if len(text) > 105 or text.endswith("."):
            continue
        candidates.append({**line, "underlined": underlined})

    candidates = _join_stacked_titles(candidates)

    # Location-aware dedupe: identical titles in two separate drawings are retained.
    seen: set[tuple[str, int, int]] = set()
    result: list[dict[str, Any]] = []
    for line in sorted(
        candidates,
        key=lambda item: (
            item.get("display_bbox", item["bbox"])[1],
            item.get("display_bbox", item["bbox"])[0],
        ),
    ):
        key_text = re.sub(r"\W+", "", line["text"].upper())
        pos_box = line.get("display_bbox", line["bbox"])
        key = (key_text, round(pos_box[0] / 15), round(pos_box[1] / 15))
        if key in seen:
            continue
        seen.add(key)
        result.append(line)
    return result[:40]


def point_to_box_dist(px: float, py: float, box: list[float]) -> float:
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0, px - x1)
    dy = max(y0 - py, 0, py - y1)
    return math.hypot(dx, dy)


def extract_callouts(
    lines: list[dict[str, Any]],
    segments: list[tuple[float, float, float, float, float]],
    width: float,
    height: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in lines:
        text = line["text"]
        upper = text.upper().strip(" .:")
        x0, y0, _, _ = line.get("display_bbox", line["bbox"])
        if y0 > 0.82 * height or x0 < 0.04 * width or len(text) < 3:
            continue
        if upper in STOP_TITLE or CALLOUT_NOISE.match(upper):
            continue

        tokenish = bool(
            re.search(
                r"(SEE|ITEM|EL\.?|ELEV|STA\.?|CONCRETE|GROUT|DRAIN|JOINT|FILLER|CAULK|SEALER|LINE|TYP|FOOTING|WALL|GIRDER|PIER|ABUT|REINF|FORM|FABRIC|MATERIAL|PATCH|CRACK|VERTICAL|OVERHEAD|GROUND|FLOW|KEYWAY|CROSSING|BRG)",
                upper,
            )
        )
        if not tokenish and len(text) > 55:
            continue

        nearest: tuple[float, list[float]] | None = None
        for xa, ya, xb, yb, _ in segments:
            distance = min(
                point_to_box_dist(xa, ya, line["bbox"]),
                point_to_box_dist(xb, yb, line["bbox"]),
            )
            if nearest is None or distance < nearest[0]:
                nearest = (distance, [round(xa, 1), round(ya, 1), round(xb, 1), round(yb, 1)])

        if nearest and nearest[0] <= 35:
            output.append(
                {
                    **line,
                    "nearest_leader_line": nearest[1],
                    "leader_distance_pt": round(nearest[0], 1),
                }
            )

    seen: set[tuple[str, int, int]] = set()
    result: list[dict[str, Any]] = []
    for callout in sorted(output, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        key = (
            callout["text"].upper(),
            round(callout["bbox"][0] / 10),
            round(callout["bbox"][1] / 10),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(callout)
    # Dense plan sheets can exceed 150 candidate labels. Truncating here caused
    # lower-page callouts to disappear before keyword filtering.
    return result


def extract_page(
    page: fitz.Page,
    page_number: int,
    *,
    textpage: fitz.TextPage | None = None,
    text_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis_box = _analysis_rect(page)
    width, height = analysis_box.width, analysis_box.height
    display_width, display_height = page.rect.width, page.rect.height
    lines = line_text_from_page(page, textpage=textpage)
    segments = collect_line_segments(page)
    sheet_title_info = extract_sheet_title_info(lines, display_width, display_height)
    sheet_title = sheet_title_info["title"]
    ocr_used = bool((text_info or {}).get("ocr_used"))
    if ocr_used:
        # OCR reliably recovers keyword positions, but its font metrics and the
        # dense table/grid geometry on flattened sheets make drawing-title and
        # leader-line classification noisy. Keep OCR hits as review mentions
        # instead of overstating them as confirmed callouts.
        drawing_titles = []
        callouts = []
    else:
        drawing_titles = extract_drawing_titles(
            lines, segments, display_width, display_height
        )
        callouts = extract_callouts(lines, segments, display_width, display_height)

    for items in (drawing_titles, callouts):
        for item in items:
            item["bbox_norm"] = norm_box(item["bbox"], width, height)

    info = text_info or {
        "text_source": "native",
        "ocr_used": False,
        "native_word_count": len(page.get_text("words")),
        "extracted_word_count": len(page.get_text("words")),
        "analysis_status": "analyzed",
        "ocr_error": None,
    }
    return {
        "page": page_number,
        "width": width,
        "height": height,
        "rotation": page.rotation,
        "display_width": display_width,
        "display_height": display_height,
        **info,
        "sheet_title": sheet_title,
        "sheet_title_lines": sheet_title_info["lines"],
        "sheet_title_bbox": sheet_title_info["bbox"],
        "sheet_title_bbox_norm": (
            norm_box(sheet_title_info["bbox"], display_width, display_height)
            if sheet_title_info["bbox"]
            else None
        ),
        "sheet_title_source": sheet_title_info["source"],
        "sheet_title_confidence": sheet_title_info["confidence"],
        "drawing_titles": drawing_titles,
        "callouts": callouts,
    }


def _harmonize_ocr_sheet_titles(pages: list[dict[str, Any]]) -> None:
    """Repair obvious OCR-only title subtitle errors using document consensus.

    Many plan sets repeat a location subtitle across multiple sheets, e.g.
    ``N. GENESEE ST / OVER / MOHAWK RIVER``. If at least two pages agree on a
    multi-line suffix and an OCR page ends with the same final line but has a
    shorter/noisy middle, preserve that page's first title line and apply the
    repeated suffix. Native-text titles are never altered.
    """
    suffix_counts: dict[tuple[str, ...], int] = {}
    suffix_original: dict[tuple[str, ...], tuple[str, ...]] = {}
    for page in pages:
        title_lines = [clean_text(value) for value in page.get("sheet_title_lines", []) if clean_text(value)]
        if len(title_lines) < 3:
            continue
        # The first line is normally the sheet-specific subject. Repeated title
        # location/context appears beneath it.
        for length in range(2, min(4, len(title_lines) - 1) + 1):
            suffix = tuple(title_lines[-length:])
            key = tuple(_compact_label(value) for value in suffix)
            suffix_counts[key] = suffix_counts.get(key, 0) + 1
            suffix_original.setdefault(key, suffix)

    repeated = [
        (len(key), count, key)
        for key, count in suffix_counts.items()
        if count >= 2
    ]
    if not repeated:
        return
    _, _, best_key = max(repeated)
    best_suffix = list(suffix_original[best_key])

    for page in pages:
        if not page.get("ocr_used"):
            continue
        current = [clean_text(value) for value in page.get("sheet_title_lines", []) if clean_text(value)]
        if not current or len(current) > len(best_suffix) + 1:
            continue
        if tuple(_compact_label(value) for value in current[-len(best_suffix):]) == best_key:
            continue
        # Require agreement on the terminal location line before correcting.
        if _compact_label(current[-1]) != best_key[-1]:
            continue
        corrected = [current[0], *best_suffix]
        page["sheet_title_lines"] = corrected
        page["sheet_title"] = " / ".join(corrected)
        page["sheet_title_confidence"] = "medium"
        page["sheet_title_source"] = (
            str(page.get("sheet_title_source", "title_block_anchors"))
            + "+document_consensus"
        )
        page["sheet_title_consensus_corrected"] = True


def extract_document(doc: fitz.Document, source_name: str) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(doc):
        textpage, text_info = _prepare_page_text(page)
        pages.append(
            extract_page(
                page,
                index + 1,
                textpage=textpage,
                text_info=text_info,
            )
        )
    _harmonize_ocr_sheet_titles(pages)
    return {"file": source_name, "pages": pages}


def extract_pdf(path: str | os.PathLike[str]) -> dict[str, Any]:
    with fitz.open(path) as doc:
        return extract_document(doc, os.path.basename(path))


def extract_pdf_bytes(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> dict[str, Any]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return extract_document(doc, filename)


def _expanded_rect(box: list[float], page_rect: fitz.Rect, margin: float = 2.5) -> fitz.Rect:
    rect = fitz.Rect(box)
    rect.x0 -= margin
    rect.y0 -= margin
    rect.x1 += margin
    rect.y1 += margin
    return rect & page_rect


def _keyword_word_rect(
    word: tuple[Any, ...], keyword: str, page_rect: fitz.Rect
) -> fitz.Rect:
    """Return a word box, extending it when the embedded text lost a glyph."""
    rect = fitz.Rect(word[:4])
    token_norm = _normalized_word(str(word[4]))
    keyword_norm = _normalized_word(keyword)
    missing = max(0, len(keyword_norm) - len(token_norm))
    # Only extend a box when the PDF text layer lost a rendered glyph. CONC.
    # is an intentional abbreviation, so highlighting should cover only it.
    if (
        missing
        and token_norm
        and _token_match_quality(token_norm, keyword_norm)
        == "recovered one-character text error"
    ):
        average_char_width = rect.width / max(1, len(token_norm))
        rect.x1 = min(page_rect.x1, rect.x1 + average_char_width * (missing + 0.35))
    return rect


def _keyword_boxes_in_callout(
    page: fitz.Page,
    box: list[float],
    keyword: str,
    textpage: fitz.TextPage | None = None,
) -> list[fitz.Rect]:
    clip = _expanded_rect(box, _analysis_rect(page), margin=3.0)
    matches: list[fitz.Rect] = []

    words = (
        page.get_text("words", clip=clip, textpage=textpage)
        if textpage
        else page.get_text("words", clip=clip)
    )
    for word in words:
        if _token_matches_keyword(str(word[4]), keyword):
            matches.append(_keyword_word_rect(word, keyword, _analysis_rect(page)))

    # Exact-search fallback for unusual span splitting / ligatures.
    if not matches:
        matches.extend(
            page.search_for(keyword, clip=clip, textpage=textpage)
            if textpage
            else page.search_for(keyword, clip=clip)
        )
    return matches



def _rect_overlap_ratio(a: list[float] | fitz.Rect, b: list[float] | fitz.Rect) -> float:
    ra, rb = fitz.Rect(a), fitz.Rect(b)
    inter = ra & rb
    if inter.is_empty:
        return 0.0
    denominator = max(1.0, min(ra.get_area(), rb.get_area()))
    return inter.get_area() / denominator


def _all_keyword_rects(
    page: fitz.Page, keyword: str, textpage: fitz.TextPage | None = None
) -> list[fitz.Rect]:
    """Return every exact or one-edit keyword occurrence on the page."""
    rects: list[fitz.Rect] = []

    words = page.get_text("words", textpage=textpage) if textpage else page.get_text("words")
    for word in words:
        if _token_matches_keyword(str(word[4]), keyword):
            rects.append(_keyword_word_rect(word, keyword, _analysis_rect(page)))

    # Native engineering PDFs can split words across spans, so search_for is a
    # useful exact fallback. OCR already supplies word-level boxes and search_for
    # can create duplicate partial boxes, so skip it for OCR text pages.
    if textpage is None:
        rects.extend(page.search_for(keyword))
    unique: list[fitz.Rect] = []
    for rect in rects:
        if any(_rect_overlap_ratio(rect, existing) >= 0.75 for existing in unique):
            continue
        unique.append(rect)
    return unique



def _nearest_text_line(
    lines: list[dict[str, Any]], rect: fitz.Rect
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    center_x = (rect.x0 + rect.x1) / 2
    center_y = (rect.y0 + rect.y1) / 2
    for line in lines:
        overlap = _rect_overlap_ratio(line["bbox"], rect)
        distance = -1000 * overlap if overlap else point_to_box_dist(center_x, center_y, line["bbox"])
        if best is None or distance < best[0]:
            best = (distance, line)
    return best[1] if best and best[0] <= 18 else None


def _horizontal_overlap_ratio(first: list[float], second: list[float]) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1.0, min(first[2] - first[0], second[2] - second[0]))


def _lines_form_stack(upper: dict[str, Any], lower: dict[str, Any]) -> bool:
    ub, lb = upper["bbox"], lower["bbox"]
    gap = lb[1] - ub[3]
    if gap < -1.5 or gap > 6.5:
        return False
    aligned = abs(ub[0] - lb[0]) <= 12.0
    overlaps = _horizontal_overlap_ratio(ub, lb) >= 0.50
    same_scale = abs(float(upper.get("font_size", 0)) - float(lower.get("font_size", 0))) <= 1.1
    return same_scale and (aligned or overlaps)


def _stacked_line_context(
    lines: list[dict[str, Any]],
    anchor_rect: fitz.Rect,
) -> tuple[str, list[float]]:
    """Return up to three tightly stacked text lines around an anchor rectangle."""
    base = _nearest_text_line(lines, anchor_rect)
    if base is None:
        return "", [anchor_rect.x0, anchor_rect.y0, anchor_rect.x1, anchor_rect.y1]

    selected = [base]
    while len(selected) < 3:
        top = selected[0]
        candidates = [line for line in lines if line not in selected and _lines_form_stack(line, top)]
        if not candidates:
            break
        candidate = min(
            candidates,
            key=lambda line: (
                top["bbox"][1] - line["bbox"][3],
                abs(top["bbox"][0] - line["bbox"][0]),
            ),
        )
        if len(candidate["text"]) + sum(len(item["text"]) for item in selected) > 160:
            break
        selected.insert(0, candidate)

    while len(selected) < 3:
        bottom = selected[-1]
        candidates = [line for line in lines if line not in selected and _lines_form_stack(bottom, line)]
        if not candidates:
            break
        candidate = min(
            candidates,
            key=lambda line: (
                line["bbox"][1] - bottom["bbox"][3],
                abs(bottom["bbox"][0] - line["bbox"][0]),
            ),
        )
        if len(candidate["text"]) + sum(len(item["text"]) for item in selected) > 160:
            break
        selected.append(candidate)

    selected.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    text = clean_text(" ".join(item["text"] for item in selected))
    bbox = [
        min(item["bbox"][0] for item in selected),
        min(item["bbox"][1] for item in selected),
        max(item["bbox"][2] for item in selected),
        max(item["bbox"][3] for item in selected),
    ]
    return text, bbox


def _keyword_line_context(
    lines: list[dict[str, Any]],
    keyword_rect: fitz.Rect,
    keyword: str,
) -> tuple[str, list[float]]:
    text, bbox = _stacked_line_context(lines, keyword_rect)
    if not text:
        return keyword, bbox
    return _repair_keyword_text(text, keyword), bbox



def _word_records(
    page: fitz.Page,
    textpage: fitz.TextPage | None = None,
    clip: fitz.Rect | None = None,
) -> list[dict[str, Any]]:
    words = (
        page.get_text("words", textpage=textpage, clip=clip)
        if textpage
        else page.get_text("words", clip=clip)
    )
    records: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        records.append(
            {
                "rect": fitz.Rect(word[:4]),
                "text": str(word[4]),
                "block": int(word[5]) if len(word) > 5 else 0,
                "line": int(word[6]) if len(word) > 6 else 0,
                "word": int(word[7]) if len(word) > 7 else index,
                "index": index,
            }
        )
    records.sort(key=lambda item: (item["block"], item["line"], item["word"], item["rect"].x0))
    return records


def _records_are_adjacent(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["block"] == second["block"]:
        if first["line"] == second["line"]:
            return 0 < second["word"] - first["word"] <= 2
        if second["line"] == first["line"] + 1 and second["word"] <= 2:
            gap = second["rect"].y0 - first["rect"].y1
            return gap <= max(first["rect"].height, second["rect"].height) * 1.8
    return False


def _keyword_occurrences(
    page: fitz.Page,
    keyword: str,
    *,
    textpage: fitz.TextPage | None = None,
    clip: fitz.Rect | None = None,
) -> list[dict[str, Any]]:
    """Find exact/fuzzy single words and contiguous multi-word phrases."""
    target_tokens = _keyword_tokens(keyword)
    if not target_tokens:
        return []
    records = _word_records(page, textpage=textpage, clip=clip)
    width = len(target_tokens)
    occurrences: list[dict[str, Any]] = []

    for index in range(len(records) - width + 1):
        window = records[index : index + width]
        if width > 1 and not all(
            _records_are_adjacent(window[offset], window[offset + 1])
            for offset in range(width - 1)
        ):
            continue
        if not all(
            _token_matches_keyword(record["text"], target)
            for record, target in zip(window, target_tokens)
        ):
            continue

        boxes = [_keyword_word_rect(
            (record["rect"].x0, record["rect"].y0, record["rect"].x1, record["rect"].y1, record["text"]),
            target,
            _analysis_rect(page),
        ) for record, target in zip(window, target_tokens)]
        union = fitz.Rect(boxes[0])
        for box in boxes[1:]:
            union |= box
        qualities = [
            _token_match_quality(record["text"], target)
            for record, target in zip(window, target_tokens)
        ]
        if all(quality == "exact" for quality in qualities):
            match_quality = "exact"
        elif any(quality == "recognized abbreviation CONC." for quality in qualities):
            match_quality = "recognized abbreviation CONC."
        else:
            match_quality = "recovered one-character text error"
        occurrences.append(
            {
                "keyword": keyword,
                "boxes": boxes,
                "bbox": union,
                "match_quality": match_quality,
            }
        )

    unique: list[dict[str, Any]] = []
    for occurrence in occurrences:
        if any(
            _rect_overlap_ratio(occurrence["bbox"], existing["bbox"]) >= 0.80
            and occurrence["keyword"].casefold() == existing["keyword"].casefold()
            for existing in unique
        ):
            continue
        unique.append(occurrence)
    return unique


def _occurrence_overlaps(
    occurrence: dict[str, Any],
    boxes: list[fitz.Rect],
    threshold: float = 0.60,
) -> bool:
    return any(
        _rect_overlap_ratio(box, existing) >= threshold
        for box in occurrence["boxes"]
        for existing in boxes
    )


def _is_obvious_non_callout(
    text: str,
    bbox: list[float],
    display_bbox: list[float],
    page_height: float,
    drawing_titles: list[dict[str, Any]],
) -> bool:
    upper = text.upper().strip()
    overlaps_drawing_title = any(
        _rect_overlap_ratio(bbox, item["bbox"]) >= 0.55 for item in drawing_titles
    )
    # The general title detector can occasionally mistake a short underlined
    # callout for a drawing title. Only use this exclusion when the text itself
    # has a strong view-title shape.
    if overlaps_drawing_title and (
        DRAWING_TITLE_WORDS.search(text) or upper.startswith("CONCRETE TABLE")
    ):
        return True
    if re.match(r"^(TABLE OF|CONCRETE TABLE|NO\.\s*-\s*INDICATES)", upper):
        return True
    # Long prose near the notes/title-block band is usually a specification note,
    # not a leader callout. Short labels in the same area are retained.
    if display_bbox[1] >= 0.78 * page_height and len(text) > 45:
        return True
    if len(text) > 115:
        return True
    return False


def _find_unclassified_keyword_mentions(
    page: fitz.Page,
    keyword: str,
    lines: list[dict[str, Any]],
    confirmed_keyword_boxes: list[fitz.Rect],
    textpage: fitz.TextPage | None = None,
) -> list[dict[str, Any]]:
    """Find keyword occurrences that strict callout detection did not classify."""
    mentions: list[dict[str, Any]] = []
    for rect in _all_keyword_rects(page, keyword, textpage=textpage):
        if any(_rect_overlap_ratio(rect, existing) >= 0.65 for existing in confirmed_keyword_boxes):
            continue
        text, bbox = _keyword_line_context(lines, rect, keyword)
        mentions.append(
            {
                "text": text,
                "bbox": [round(v, 2) for v in bbox],
                "keyword_box": [round(v, 2) for v in (rect.x0, rect.y0, rect.x1, rect.y1)],
                "classification": "review",
            }
        )
    return mentions



def highlight_keywords_callouts_bytes(
    pdf_bytes: bytes,
    keywords: Iterable[str] | str,
    filename: str = "uploaded.pdf",
    include_review_mentions: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Highlight configured keywords while suppressing known non-555 concrete work."""
    keyword_list = normalize_keywords(keywords)
    if not keyword_list:
        raise ValueError("Provide at least one keyword or phrase.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_report: list[dict[str, Any]] = []
    seen_annotation_boxes: set[tuple[int, int, int, int, int]] = set()
    counts = {
        keyword: {
            "confirmed_callouts": 0,
            "review_mentions": 0,
            "excluded_mentions": 0,
            "highlighted_mentions": 0,
        }
        for keyword in keyword_list
    }

    try:
        for page_index, page in enumerate(doc):
            textpage, text_info = _prepare_page_text(page)
            extracted = extract_page(
                page,
                page_index + 1,
                textpage=textpage,
                text_info=text_info,
            )
            lines = line_text_from_page(page, textpage=textpage)
            analysis_box = _analysis_rect(page)
            confirmed_boxes: list[fitz.Rect] = []
            excluded_boxes: list[fitz.Rect] = []
            flagged: list[dict[str, Any]] = []
            excluded_mentions: list[dict[str, Any]] = []
            excluded_keys: set[tuple[str, int, int, int, int]] = set()

            def record_excluded(
                *,
                keyword: str,
                text: str,
                bbox: fitz.Rect | list[float],
                occurrences: list[dict[str, Any]],
                terms: list[str],
                source: str,
            ) -> None:
                rect = fitz.Rect(bbox)
                key = (
                    keyword.casefold(),
                    round(rect.x0),
                    round(rect.y0),
                    round(rect.x1),
                    round(rect.y1),
                )
                if key in excluded_keys:
                    return
                excluded_keys.add(key)
                keyword_boxes: list[list[float]] = []
                match_quality = "exact"
                for occurrence in occurrences:
                    if occurrence.get("match_quality") != "exact":
                        match_quality = occurrence.get("match_quality", match_quality)
                    excluded_boxes.extend(occurrence.get("boxes", []))
                    for box in occurrence.get("boxes", []):
                        keyword_boxes.append(
                            [round(v, 2) for v in (box.x0, box.y0, box.x1, box.y1)]
                        )
                excluded_mentions.append(
                    {
                        "text": text,
                        "bbox": [round(v, 2) for v in rect],
                        "bbox_norm": norm_box(rect, analysis_box.width, analysis_box.height),
                        "keyword_boxes": keyword_boxes,
                        "matched_keywords": [keyword],
                        "match_details": [
                            {"keyword": keyword, "match_quality": match_quality}
                        ],
                        "classification": "excluded non-555 concrete mention",
                        "exclusion_terms": terms,
                        "exclusion_reason": "Concrete appears with a known non-Item-555 term",
                        "source_classification": source,
                    }
                )
                counts[keyword]["excluded_mentions"] += 1

            for callout in extracted["callouts"]:
                context_text, context_bbox = _stacked_line_context(
                    lines, fitz.Rect(callout["bbox"])
                )
                if not context_text:
                    context_text, context_bbox = callout["text"], callout["bbox"]

                matched: list[tuple[str, list[dict[str, Any]]]] = []
                for keyword in keyword_list:
                    if not _text_contains_keyword(context_text, keyword):
                        continue
                    clip = _expanded_rect(context_bbox, analysis_box, margin=4.0)
                    occurrences = _keyword_occurrences(
                        page,
                        keyword,
                        textpage=textpage,
                        clip=clip,
                    )
                    if occurrences:
                        matched.append((keyword, occurrences))

                if not matched:
                    continue
                repaired_text = context_text
                for keyword, _ in matched:
                    repaired_text = _repair_keyword_text(repaired_text, keyword)
                if _is_obvious_non_callout(
                    repaired_text,
                    context_bbox,
                    callout.get("display_bbox", callout["bbox"]),
                    extracted["display_height"],
                    extracted["drawing_titles"],
                ):
                    continue

                matched_keywords: list[str] = []
                match_details: list[dict[str, Any]] = []
                highlighted_boxes: list[list[float]] = []
                for keyword, occurrences in matched:
                    # Latest product rule: CONC. / conc. must always be surfaced,
                    # including in sheet titles and even when nearby words (for
                    # example DECK) are on Carol's non-555 exclusion list. Exact
                    # CONCRETE spellings continue to use the exclusion classifier.
                    abbreviation_occurrences = [
                        occurrence
                        for occurrence in occurrences
                        if occurrence.get("match_quality")
                        == "recognized abbreviation CONC."
                    ]
                    standard_occurrences = [
                        occurrence
                        for occurrence in occurrences
                        if occurrence.get("match_quality")
                        != "recognized abbreviation CONC."
                    ]

                    exclusion_terms = (
                        _concrete_exclusion_terms(repaired_text)
                        if _keyword_is_concrete(keyword) and standard_occurrences
                        else []
                    )
                    if exclusion_terms:
                        record_excluded(
                            keyword=keyword,
                            text=repaired_text,
                            bbox=context_bbox,
                            occurrences=standard_occurrences,
                            terms=exclusion_terms,
                            source="confirmed callout",
                        )

                    allowed_occurrences = (
                        abbreviation_occurrences
                        if exclusion_terms
                        else occurrences
                    )
                    if not allowed_occurrences:
                        continue

                    keyword_was_highlighted = False
                    keyword_quality = "exact"
                    for occurrence in allowed_occurrences:
                        if occurrence["match_quality"] != "exact":
                            keyword_quality = occurrence["match_quality"]
                        confirmed_boxes.extend(occurrence["boxes"])
                        for rect in occurrence["boxes"]:
                            key = (
                                page_index,
                                round(rect.x0 * 2),
                                round(rect.y0 * 2),
                                round(rect.x1 * 2),
                                round(rect.y1 * 2),
                            )
                            if key in seen_annotation_boxes:
                                continue
                            seen_annotation_boxes.add(key)
                            annotation = page.add_highlight_annot(rect)
                            annotation.set_info(
                                title=f"{keyword} callout",
                                content=repaired_text,
                            )
                            annotation.update()
                            highlighted_boxes.append(
                                [round(v, 2) for v in (rect.x0, rect.y0, rect.x1, rect.y1)]
                            )
                            keyword_was_highlighted = True
                    if keyword_was_highlighted:
                        matched_keywords.append(keyword)
                        match_details.append(
                            {"keyword": keyword, "match_quality": keyword_quality}
                        )
                        counts[keyword]["confirmed_callouts"] += 1
                        counts[keyword]["highlighted_mentions"] += 1

                if highlighted_boxes:
                    flagged.append(
                        {
                            "text": repaired_text,
                            "bbox": [round(v, 2) for v in context_bbox],
                            "bbox_norm": norm_box(
                                context_bbox, analysis_box.width, analysis_box.height
                            ),
                            "keyword_boxes": highlighted_boxes,
                            "matched_keywords": matched_keywords,
                            "match_details": match_details,
                            "classification": "confirmed callout",
                        }
                    )

            review_mentions: list[dict[str, Any]] = []
            for keyword in keyword_list:
                for occurrence in _keyword_occurrences(
                    page,
                    keyword,
                    textpage=textpage,
                ):
                    if _occurrence_overlaps(occurrence, confirmed_boxes + excluded_boxes):
                        continue
                    # Use stacked local context so exclusions split across two CAD text
                    # lines (for example CONCRETE on one line and CURB on the next)
                    # are still evaluated as one callout phrase.
                    context_text, context_bbox = _stacked_line_context(
                        lines, occurrence["bbox"]
                    )
                    if not context_text:
                        context_text, context_bbox = _keyword_line_context(
                            lines, occurrence["bbox"], keyword
                        )
                    repaired_text = _repair_keyword_text(context_text, keyword)
                    # CONC. / conc. is an explicit always-highlight alias.
                    # Do not suppress it via the non-555 exclusion list.
                    is_conc_abbreviation = (
                        occurrence.get("match_quality")
                        == "recognized abbreviation CONC."
                    )
                    exclusion_terms = (
                        _concrete_exclusion_terms(repaired_text)
                        if _keyword_is_concrete(keyword) and not is_conc_abbreviation
                        else []
                    )
                    if exclusion_terms:
                        record_excluded(
                            keyword=keyword,
                            text=repaired_text,
                            bbox=context_bbox,
                            occurrences=[occurrence],
                            terms=exclusion_terms,
                            source="review mention",
                        )
                        continue

                    review_mentions.append(
                        {
                            "text": repaired_text,
                            "bbox": [round(v, 2) for v in context_bbox],
                            "bbox_norm": norm_box(
                                context_bbox, analysis_box.width, analysis_box.height
                            ),
                            "keyword_boxes": [
                                [round(v, 2) for v in (box.x0, box.y0, box.x1, box.y1)]
                                for box in occurrence["boxes"]
                            ],
                            "matched_keywords": [keyword],
                            "match_details": [
                                {
                                    "keyword": keyword,
                                    "match_quality": occurrence["match_quality"],
                                }
                            ],
                            "classification": "review mention",
                        }
                    )
                    counts[keyword]["review_mentions"] += 1

            if include_review_mentions:
                grouped_review: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
                for mention in review_mentions:
                    keyword = mention["matched_keywords"][0]
                    any_new_box = False
                    for raw_box in mention["keyword_boxes"]:
                        rect = fitz.Rect(raw_box)
                        key = (
                            page_index,
                            round(rect.x0 * 2),
                            round(rect.y0 * 2),
                            round(rect.x1 * 2),
                            round(rect.y1 * 2),
                        )
                        if key in seen_annotation_boxes:
                            continue
                        seen_annotation_boxes.add(key)
                        annotation = page.add_highlight_annot(rect)
                        annotation.set_info(
                            title=f"{keyword} mention — review",
                            content=mention["text"],
                        )
                        annotation.update()
                        any_new_box = True
                    if not any_new_box:
                        continue
                    box = mention["bbox"]
                    group_key = (
                        mention["text"],
                        round(box[0]), round(box[1]), round(box[2]), round(box[3]),
                    )
                    if group_key not in grouped_review:
                        grouped_review[group_key] = dict(mention)
                    else:
                        grouped = grouped_review[group_key]
                        if keyword not in grouped["matched_keywords"]:
                            grouped["matched_keywords"].append(keyword)
                            grouped["match_details"].extend(mention["match_details"])
                            grouped["keyword_boxes"].extend(mention["keyword_boxes"])
                    counts[keyword]["highlighted_mentions"] += 1
                flagged.extend(grouped_review.values())

            pages_report.append(
                {
                    "page": page_index + 1,
                    "analysis_status": extracted["analysis_status"],
                    "text_source": extracted["text_source"],
                    "ocr_used": extracted["ocr_used"],
                    "native_word_count": extracted["native_word_count"],
                    "extracted_word_count": extracted["extracted_word_count"],
                    "ocr_error": extracted.get("ocr_error"),
                    "rotation": extracted["rotation"],
                    "sheet_title": extracted["sheet_title"],
                    "sheet_title_lines": extracted.get("sheet_title_lines", []),
                    "sheet_title_bbox": extracted.get("sheet_title_bbox"),
                    "sheet_title_bbox_norm": extracted.get("sheet_title_bbox_norm"),
                    "sheet_title_source": extracted.get("sheet_title_source"),
                    "sheet_title_confidence": extracted.get("sheet_title_confidence"),
                    "sheet_title_consensus_corrected": False,
                    "drawing_titles": [item["text"] for item in extracted["drawing_titles"]],
                    "flagged_callouts": flagged,
                    "other_keyword_mentions": review_mentions,
                    "excluded_non_555_mentions": excluded_mentions,
                }
            )

        _harmonize_ocr_sheet_titles(pages_report)
        output_bytes = doc.tobytes(garbage=4, deflate=True)
    finally:
        doc.close()

    report = {
        "file": filename,
        "keywords": keyword_list,
        "concrete_non_555_exclusions": CONCRETE_NON_555_EXCLUSIONS,
        "total_pages": len(pages_report),
        "pages_analyzed": sum(
            1 for page in pages_report if page.get("analysis_status") != "limited: OCR unavailable"
        ),
        "ocr_pages": [page["page"] for page in pages_report if page.get("ocr_used")],
        "limited_pages": [
            page["page"]
            for page in pages_report
            if page.get("analysis_status") == "limited: OCR unavailable"
        ],
        "total_flagged_callouts": sum(
            sum(
                1 for item in page["flagged_callouts"]
                if item.get("classification") == "confirmed callout"
            )
            for page in pages_report
        ),
        "total_review_mentions": sum(
            len(page.get("other_keyword_mentions", [])) for page in pages_report
        ),
        "total_excluded_non_555_mentions": sum(
            len(page.get("excluded_non_555_mentions", [])) for page in pages_report
        ),
        "total_highlighted_mentions": sum(
            len(page.get("flagged_callouts", [])) for page in pages_report
        ),
        "total_recovered_matches": sum(
            sum(
                1
                for item in page.get("flagged_callouts", [])
                for detail in item.get("match_details", [])
                if detail.get("match_quality") == "recovered one-character text error"
            )
            for page in pages_report
        ),
        "keyword_counts": counts,
        "include_review_mentions": include_review_mentions,
        "pages": pages_report,
    }
    return output_bytes, report


def highlight_keyword_callouts_bytes(
    pdf_bytes: bytes,
    keyword: str = "concrete",
    filename: str = "uploaded.pdf",
    include_review_mentions: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Backward-compatible single-keyword wrapper."""
    return highlight_keywords_callouts_bytes(
        pdf_bytes,
        [keyword],
        filename=filename,
        include_review_mentions=include_review_mentions,
    )


def render_pdf_page_bytes(pdf_bytes: bytes, page_number: int, zoom: float = 1.35) -> bytes:
    """Render one page (1-indexed) to PNG for the MVP preview."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"Page {page_number} is outside 1-{len(doc)}")
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, annots=True)
        return pixmap.tobytes("png")


def iter_pdfs(inputs: Iterable[str]) -> Iterable[str]:
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            yield from sorted(str(candidate) for candidate in path.glob("*.pdf"))
        elif path.suffix.lower() == ".pdf":
            yield str(path)


def write_csv(results: list[dict[str, Any]], csv_path: str | os.PathLike[str]) -> None:
    rows: list[dict[str, str | int]] = []
    for result in results:
        for page in result["pages"]:
            rows.append(
                {
                    "file": result["file"],
                    "page": page["page"],
                    "type": "sheet_title",
                    "text": page.get("sheet_title") or "",
                    "bbox_norm": "",
                }
            )
            for drawing_title in page["drawing_titles"]:
                rows.append(
                    {
                        "file": result["file"],
                        "page": page["page"],
                        "type": "drawing_title",
                        "text": drawing_title["text"],
                        "bbox_norm": json.dumps(drawing_title["bbox_norm"]),
                    }
                )
            for callout in page["callouts"]:
                rows.append(
                    {
                        "file": result["file"],
                        "page": page["page"],
                        "type": "callout",
                        "text": callout["text"],
                        "bbox_norm": json.dumps(callout["bbox_norm"]),
                    }
                )

    with open(csv_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["file", "page", "type", "text", "bbox_norm"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="PDF files or folders")
    parser.add_argument("-o", "--output", default="extracted_plan_data.json")
    parser.add_argument("--csv", default=None)
    parser.add_argument(
        "--highlight-keyword",
        default=None,
        help='Highlight this word only when it appears in a detected callout (e.g. "concrete").',
    )
    parser.add_argument(
        "--highlight-dir",
        default="highlighted",
        help="Output directory for highlighted PDFs and reports.",
    )
    args = parser.parse_args()

    pdfs = list(iter_pdfs(args.inputs))
    if not pdfs:
        raise SystemExit("No PDFs found")

    results = [extract_pdf(path) for path in pdfs]
    with open(args.output, "w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2)
    if args.csv:
        write_csv(results, args.csv)

    if args.highlight_keyword:
        highlight_dir = Path(args.highlight_dir)
        highlight_dir.mkdir(parents=True, exist_ok=True)
        for path in pdfs:
            source_bytes = Path(path).read_bytes()
            highlighted_bytes, report = highlight_keyword_callouts_bytes(
                source_bytes,
                keyword=args.highlight_keyword,
                filename=os.path.basename(path),
            )
            stem = Path(path).stem
            (highlight_dir / f"{stem}_highlighted.pdf").write_bytes(highlighted_bytes)
            (highlight_dir / f"{stem}_report.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

    message = f"Wrote {args.output}"
    if args.csv:
        message += f" and {args.csv}"
    if args.highlight_keyword:
        message += f"; highlighted PDFs are in {args.highlight_dir}"
    print(message)


if __name__ == "__main__":
    main()
