"""Small keyword-matching regression test for the plan review MVP.

Run with:
    python regression_test.py

The test intentionally wraps WIRE / FABRIC and the long patching-material phrase
across separate PDF text blocks. It also verifies the concrete exclusion, Item
555 override rules, and that adjacent quantity-table rows are never merged into
one finding.
"""

from __future__ import annotations

import fitz

from pdf_plan_extractor import (
    _apply_cross_sheet_item_555_rule,
    highlight_keywords_callouts_bytes,
)

KEYWORDS = [
    "Concrete",
    "Grout",
    "Vertical and Overhead Patching Material",
    "Joint Filler",
    "Caulking Compound",
    "Joint Sealer",
    "Wire Fabric",
    "Form",
    "Forms",
    "Crack Repairs",
    "Cast in Place",
    "CIP.",
]


def build_test_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page(width=900, height=780)

    rows = [
        (60, 70, ["CONCRETE ITEM 555.0021"]),
        (60, 110, ["GROUT"]),
        (60, 150, ["VERTICAL AND OVERHEAD", "PATCHING MATERIAL"]),
        (60, 215, ["JOINT FILLER"]),
        (60, 255, ["CAULKING COMPOUND"]),
        (60, 295, ["JOINT SEALER"]),
        (60, 335, ["EPOXY COATED WELDED WIRE", "FABRIC REINFORCEMENT"]),
        (60, 400, ["FORM"]),
        (60, 440, ["FORMS"]),
        (60, 480, ["CRACK REPAIRS"]),
        (60, 500, ["CAST IN PLACE FOUNDATION"]),
        (60, 520, ["CAST-IN-PLACE FOUNDATION"]),
        (60, 540, ["CIP. FOUNDATION"]),
        (60, 570, ["LONGITUDINAL SAWCUT GROOVING OF STRUCTURAL SLAB"]),
        (60, 590, ["PROTECTIVE SEALING OF STRUCTURAL CONCRETE"]),
        (60, 610, ["CAST-IN-PLACE CONCRETE CURB VF150"]),
    ]

    for x, y, lines in rows:
        for offset, line in enumerate(lines):
            # Each wrapped line is inserted separately to mimic CAD exporters
            # that split one visible callout across independent text blocks.
            page.insert_text((x, y + offset * 14), line, fontsize=10)
        page.draw_line((x + 280, y - 2), (x + 335, y - 20))

    # Quantity-table regression: these are three separate rows, not one wrapped
    # annotation.  The old 6.5-point stacking tolerance incorrectly merged all
    # three rows into the Concrete finding.
    table_x = 500
    table_y = 670
    table_rows = [
        "STONE FILLING (LIGHT)",
        "BEDDING MATERIAL, TYPE 1",
        "CONCRETE CYLINDER CURING EQUIPMENT",
    ]
    for index, row in enumerate(table_rows):
        baseline = table_y + index * 11.25
        page.insert_text((table_x, baseline), row, fontsize=5)
        page.draw_line((table_x - 8, baseline + 2.5), (820, baseline + 2.5))

    payload = document.tobytes()
    document.close()
    return payload



def _build_cross_sheet_rule_fixture(*, include_item_555: bool) -> tuple[bytes, dict]:
    """Build a minimal repeated-title family for cross-sheet 555 testing."""
    document = fitz.open()
    page1 = document.new_page(width=800, height=600)
    phrase = (
        "REINFORCEMENT SHOWN FOR CAST IN PLACE CONCRETE. "
        "CONTRACTOR MAY ELECT TO INSTALL AS PRECAST CONCRETE."
    )
    page1.insert_text((80, 180), phrase, fontsize=10)
    concrete_boxes = [list(rect) for rect in page1.search_for("CONCRETE")]

    page2 = document.new_page(width=800, height=600)
    page2.insert_text((80, 180), "RELATED DETAIL", fontsize=10)
    if include_item_555:
        page2.insert_text((80, 220), "ITEM 555.0021", fontsize=10)

    payload = document.tobytes()
    document.close()
    first_box = concrete_boxes[0]
    report = {
        "pages": [
            {
                "page": 1,
                "sheet_title": "TRUNK LINE OVER EBSS / DETAILS / SHEET 1 OF 2",
                "sheet_title_lines": ["TRUNK LINE OVER EBSS", "DETAILS", "SHEET 1 OF 2"],
                "flagged_callouts": [],
                "excluded_non_555_mentions": [
                    {
                        "text": phrase,
                        "bbox": first_box,
                        "bbox_norm": first_box,
                        "keyword_boxes": concrete_boxes,
                        "matched_keywords": ["Concrete"],
                        "match_details": [{"keyword": "Concrete", "match_quality": "exact"}],
                        "classification": "excluded non-555 concrete mention",
                        "exclusion_terms": ["Reinforcement", "Precast"],
                        "exclusion_reason": "Concrete appears with a known non-Item-555 term",
                        "source_classification": "review mention",
                    }
                ],
            },
            {
                "page": 2,
                "sheet_title": "TRUNK LINE OVER EBSS / DETAILS / SHEET 2 OF 2",
                "sheet_title_lines": ["TRUNK LINE OVER EBSS", "DETAILS", "SHEET 2 OF 2"],
                "flagged_callouts": [],
                "excluded_non_555_mentions": [],
            },
        ],
        "keyword_counts": {
            "Concrete": {
                "confirmed_callouts": 0,
                "review_mentions": 0,
                "excluded_mentions": 1,
                "highlighted_mentions": 0,
            }
        },
    }
    return payload, report


def _assert_cross_sheet_rule() -> None:
    for include_item_555, expected_classification in (
        (False, "cross-sheet missing Item 555"),
        (True, "cross-sheet Item 555 override"),
    ):
        payload, report = _build_cross_sheet_rule_fixture(
            include_item_555=include_item_555
        )
        highlighted, updated = _apply_cross_sheet_item_555_rule(
            payload, payload, report
        )
        assert highlighted
        moved = updated["pages"][0]["flagged_callouts"]
        assert len(moved) == 1, moved
        finding = moved[0]
        assert finding["classification"] == expected_classification, finding
        assert set(finding["overridden_exclusions"]) == {
            "Reinforcement",
            "Precast",
        }
        assert not updated["pages"][0]["excluded_non_555_mentions"]
        evidence = finding["cross_sheet_item_555"]
        assert evidence["related_pages"] == [1, 2]
        if include_item_555:
            assert evidence["item_555_pages"] == [2]
        else:
            assert evidence["item_555_pages"] == []


def main() -> None:
    highlighted, report = highlight_keywords_callouts_bytes(
        build_test_pdf(),
        KEYWORDS,
        filename="keyword-regression.pdf",
        include_review_mentions=True,
    )
    assert highlighted
    assert report["total_unresolved_keyword_occurrences"] == 0, report["keyword_coverage"]

    for keyword in KEYWORDS:
        detected = report["keyword_coverage"][keyword]["detected_occurrences"]
        assert detected >= 1, f"Keyword was not detected: {keyword}"

    wire_count = report["keyword_coverage"]["Wire Fabric"]["detected_occurrences"]
    assert wire_count == 1, f"Expected one wrapped Wire Fabric occurrence, got {wire_count}"

    exclusions = [
        item
        for page in report["pages"]
        for item in page.get("excluded_non_555_mentions", [])
    ]
    assert any("Curb" in item.get("exclusion_terms", []) for item in exclusions)

    flagged_concrete = [
        item
        for page in report["pages"]
        for item in page.get("flagged_callouts", [])
        if "Concrete" in item.get("matched_keywords", [])
    ]
    assert any("555.0021" in item.get("text", "") for item in flagged_concrete)


    assert any(
        "Protective sealing" in item.get("exclusion_terms", [])
        for item in exclusions
    ), exclusions
    assert not any(
        "PROTECTIVE SEALING" in item.get("text", "").upper()
        for item in [
            finding
            for page in report["pages"]
            for finding in page.get("flagged_callouts", [])
        ]
    )

    # The text shown in All Flagged must describe the exact keyword box that was
    # highlighted.  Every reported keyword must occur in its own finding text,
    # and every annotation popup must use that same text.
    for page in report["pages"]:
        for finding in page.get("flagged_callouts", []):
            upper = finding.get("text", "").upper()
            normalized_text = "".join(ch for ch in upper if ch.isalnum())
            for keyword in finding.get("matched_keywords", []):
                normalized_keyword = "".join(ch for ch in keyword.upper() if ch.isalnum())
                assert (
                    normalized_keyword in normalized_text
                    or (keyword == "Concrete" and "CONC" in normalized_text)
                ), finding
            assert finding.get("keyword_boxes"), finding

    annotated = fitz.open(stream=highlighted, filetype="pdf")
    annotation_texts = {
        annotation.info.get("content", "")
        for page in annotated
        for annotation in (page.annots() or [])
    }
    annotated.close()
    flagged_texts = {
        finding.get("text", "")
        for page in report["pages"]
        for finding in page.get("flagged_callouts", [])
    }
    assert annotation_texts <= flagged_texts, (annotation_texts - flagged_texts)

    all_flagged_text = [
        item.get("text", "")
        for page in report["pages"]
        for item in page.get("flagged_callouts", [])
    ]
    cylinder_rows = [text for text in all_flagged_text if "CONCRETE CYLINDER" in text]
    assert cylinder_rows == ["CONCRETE CYLINDER CURING EQUIPMENT"], cylinder_rows
    assert not any(
        "STONE FILLING" in text and "CONCRETE CYLINDER" in text
        for text in all_flagged_text
    ), all_flagged_text

    cast_in_place_count = report["keyword_coverage"]["Cast in Place"]["detected_occurrences"]
    assert cast_in_place_count == 3, (
        f"Expected spaced and both hyphenated Cast in Place matches, got {cast_in_place_count}"
    )
    cip_count = report["keyword_coverage"]["CIP."]["detected_occurrences"]
    assert cip_count == 1, f"Expected one CIP. occurrence, got {cip_count}"

    _assert_cross_sheet_rule()

    print("PASS: keyword coverage, exclusions, highlight/report consistency, wrapped phrases, table-row isolation, Cast in Place/CIP variants, and cross-sheet Item 555 logic verified.")


if __name__ == "__main__":
    main()
