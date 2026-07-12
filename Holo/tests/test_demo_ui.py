"""UI contracts for the demo page that must survive live, unbounded model catalogs."""

from __future__ import annotations

import re

from plva_proxy import demo


def _ui_html() -> str:
    return demo.UI_PATH.read_text("utf-8")


def test_model_menu_scrolls_when_catalog_is_long() -> None:
    html = _ui_html()
    rules = re.findall(r"\.modelmenu\{[^}]*\}", html)
    assert rules, "missing .modelmenu rule"
    assert any("max-height" in rule and "overflow-y:auto" in rule for rule in rules)


def test_lab_model_control_is_a_select_not_a_segmented_control() -> None:
    html = _ui_html()
    assert 'id="lab-model-select"' in html
    assert 'segRow("Model"' not in html
