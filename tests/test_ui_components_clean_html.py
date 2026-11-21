"""Regression tests verifying UI components render clean HTML without code-block artifacts.

In CommonMark/Markdown, any line starting with 4 or more spaces is treated as an
indented code block (<pre><code>). Blank lines inside HTML can also cause Markdown
parsers to terminate HTML mode prematurely and treat subsequent HTML as raw text.

These tests ensure:
1. `clean_html` strips all indentation and filters out empty lines.
2. Every HTML-generating helper in `dashboard.ui.components` produces markup with zero
   lines indented by 4+ spaces and zero blank lines.
3. Every rendering component (kpi_card, page_header, insight_banner, etc.) delivers
   clean HTML to Streamlit without code-block formatting.
4. AppTest on `model_performance.py` contains no accidental code blocks displaying
   raw HTML markup like '</DIV>' or 'KPI-VALUE'.
"""
from __future__ import annotations

import re
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from streamlit.testing.v1 import AppTest

from dashboard.ui.components import (
    clean_html,
    render_html,
    status_pill,
    data_badge,
    top_bar_system_status,
    risk_chip,
    risk_gauge_svg,
    render_in_cell_bar,
    apply_page_styles,
    sidebar_brand,
    page_header,
    section_header,
    kpi_card,
    insight_banner,
    empty_state,
)

BASE_DIR = Path(__file__).resolve().parents[1]
MP_PATH = str(BASE_DIR / "dashboard" / "pages" / "model_performance.py")


class TestUIComponentsCleanHtml(unittest.TestCase):
    """Verify that all UI components render clean HTML without code-block artifacts."""

    def test_clean_html_strips_indentation_and_blank_lines(self):
        raw_html = """
            <div class="kpi-card">
                <div class="kpi-label">METRIC</div>

                <div class="kpi-value">95.4%</div>
            </div>
        """
        cleaned = clean_html(raw_html)
        lines = cleaned.splitlines()

        # No empty lines
        self.assertTrue(all(line.strip() != "" for line in lines))
        # No lines with 4+ leading spaces
        self.assertTrue(all(not line.startswith("    ") for line in lines))
        # No leading or trailing whitespace on any line
        self.assertTrue(all(line == line.strip() for line in lines))

    def test_string_components_contain_no_4_plus_spaces_or_blank_lines(self):
        """Assert all string-returning components produce zero 4+ indented or blank lines."""
        components = [
            status_pill("LIVE", "live"),
            status_pill("DEGRADED", "warn"),
            status_pill("OFFLINE", "neutral"),
            data_badge("SYNTHETIC DATASET"),
            top_bar_system_status(model_loaded=True, data_loaded=True, total_rows=10000),
            risk_chip("High"),
            risk_chip("Medium"),
            risk_chip("Low"),
            risk_gauge_svg(0.42, "High"),
            render_in_cell_bar(75.5, 0.0, 100.0),
        ]

        for comp_output in components:
            self.assertIsInstance(comp_output, str)
            self.assertGreater(len(comp_output), 0)
            lines = comp_output.splitlines()
            for line_idx, line in enumerate(lines):
                self.assertFalse(
                    line.startswith("    "),
                    f"Line {line_idx} starts with 4+ spaces in:\n{comp_output}",
                )
                self.assertNotEqual(
                    line.strip(),
                    "",
                    f"Blank line found at index {line_idx} in:\n{comp_output}",
                )

    def test_rendering_components_call_render_html_cleanly(self):
        """Assert all rendering components route through render_html with valid markup."""
        # Test render_html itself with mock container
        mock_container = MagicMock()
        indented_input = "    <div>\n\n        <span>Hello</span>\n    </div>"
        render_html(indented_input, container=mock_container)
        mock_container.markdown.assert_called_once()
        called_arg, called_kwargs = mock_container.markdown.call_args
        self.assertEqual(called_kwargs.get("unsafe_allow_html"), True)
        self.assertNotIn("    ", called_arg[0])
        self.assertNotIn("\n\n", called_arg[0])

        # Test apply_page_styles
        with patch("dashboard.ui.components.render_html") as mock_render:
            from dashboard.ui import components as _ui
            _ui.apply_page_styles()
            mock_render.assert_called_once()
            css_arg = mock_render.call_args[0][0]
            self.assertIn("<style>", css_arg)

        # Test sidebar_brand
        with patch("dashboard.ui.components.render_html") as mock_render:
            from dashboard.ui import components as _ui
            _ui.sidebar_brand()
            mock_render.assert_called_once()
            html_arg = mock_render.call_args[0][0]
            cleaned = clean_html(html_arg)
            for line in cleaned.splitlines():
                self.assertFalse(line.startswith("    "))
                self.assertNotEqual(line.strip(), "")

        # Test page_header
        with patch("dashboard.ui.components.render_html") as mock_render:
            from dashboard.ui import components as _ui
            _ui.page_header("Test Title", "Test Subtitle")
            mock_render.assert_called_once()
            html_arg = mock_render.call_args[0][0]
            cleaned = clean_html(html_arg)
            for line in cleaned.splitlines():
                self.assertFalse(line.startswith("    "))
                self.assertNotEqual(line.strip(), "")

        # Test section_header
        with patch("dashboard.ui.components.render_html") as mock_render:
            from dashboard.ui import components as _ui
            _ui.section_header("Test Section", "Subtitle")
            mock_render.assert_called_once()
            html_arg = mock_render.call_args[0][0]
            cleaned = clean_html(html_arg)
            for line in cleaned.splitlines():
                self.assertFalse(line.startswith("    "))
                self.assertNotEqual(line.strip(), "")

        # Test kpi_card
        with patch("dashboard.ui.components.render_html") as mock_render:
            from dashboard.ui import components as _ui
            _ui.kpi_card(
                label="BRIER LOSS SCORE",
                value="0.1082",
                delta="-0.0210",
                delta_basis="calibrated",
                delta_direction="positive",
            )
            mock_render.assert_called_once()
            html_arg = mock_render.call_args[0][0]
            cleaned = clean_html(html_arg)
            for line in cleaned.splitlines():
                self.assertFalse(line.startswith("    "), f"kpi_card emitted 4+ leading spaces: {line}")
                self.assertNotEqual(line.strip(), "", "kpi_card emitted blank line")

        # Test insight_banner
        with patch("dashboard.ui.components.render_html") as mock_render:
            from dashboard.ui import components as _ui
            _ui.insight_banner(
                title="Model Calibration",
                text="Calibration improved Brier loss from 0.1292 to 0.1082 on test split.",
                badge="MODEL GOVERNANCE",
                icon="⚡",
            )
            mock_render.assert_called_once()
            html_arg = mock_render.call_args[0][0]
            cleaned = clean_html(html_arg)
            for line in cleaned.splitlines():
                self.assertFalse(line.startswith("    "))
                self.assertNotEqual(line.strip(), "")

        # Test empty_state
        with patch("dashboard.ui.components.render_html") as mock_render:
            from dashboard.ui import components as _ui
            _ui.empty_state("No data matches current filter settings")
            mock_render.assert_called_once()
            html_arg = mock_render.call_args[0][0]
            cleaned = clean_html(html_arg)
            for line in cleaned.splitlines():
                self.assertFalse(line.startswith("    "))
                self.assertNotEqual(line.strip(), "")

    def test_model_performance_rendered_markdown_has_no_raw_html_code_blocks(self):
        """Verify AppTest on model_performance.py does not render raw HTML tags as text or code."""
        at = AppTest.from_file(MP_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0, f"model_performance raised exceptions: {at.exception}")

        # Check all markdown elements rendered on the page
        for idx, md in enumerate(at.markdown):
            val = md.value
            # If the block is an HTML block, ensure no line has 4+ leading spaces or blank lines
            if val.strip().startswith("<"):
                lines = val.splitlines()
                for line_idx, line in enumerate(lines):
                    self.assertFalse(
                        line.startswith("    "),
                        f"Markdown element {idx} line {line_idx} has 4+ leading spaces:\n{line}",
                    )
                    self.assertNotEqual(
                        line.strip(),
                        "",
                        f"Markdown element {idx} line {line_idx} has a blank line in HTML",
                    )
            # Ensure raw escaped uppercase tags like '</DIV>' or leaked 'KPI-VALUE' as uppercase code blocks don't exist
            self.assertNotIn("</DIV>", val, f"Raw uppercase </DIV> found in markdown value: {val[:200]}")
            self.assertNotIn("KPI-VALUE", val, f"Raw uppercase KPI-VALUE code block found: {val[:200]}")


if __name__ == "__main__":
    unittest.main()
