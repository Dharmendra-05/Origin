"""
Visualizer module for generating line-level attribution heatmaps in HTML and ASCII formats.
"""

from typing import List
from origin_rag.chunker import TextChunk
from origin_rag.attribution import AttributionReport


class AttributionVisualizer:
    """Generates visual heatmaps highlighting grounded vs ungrounded text lines."""

    @staticmethod
    def render_ascii_heatmap(report: AttributionReport, chunk: TextChunk) -> str:
        """Renders an ASCII text heatmap of document line citations."""
        lines = chunk.content.splitlines()
        output = [f"=== LINE ATTRIBUTION HEATMAP: {chunk.file_name} (#L{chunk.start_line}-L{chunk.end_line}) ==="]

        matched_lines = set()
        for cite in report.citations:
            if cite.source_file == chunk.file_name:
                for l_num in range(cite.start_line, cite.end_line + 1):
                    matched_lines.add(l_num)

        for idx, line in enumerate(lines, chunk.start_line):
            indicator = "[✓ MATCH]" if idx in matched_lines else "[  PASS ]"
            output.append(f"{indicator} L{idx:03d}: {line}")

        return "\n".join(output)

    @staticmethod
    def render_html_heatmap(report: AttributionReport, chunk: TextChunk) -> str:
        """Renders an HTML snippet with line-level background color coding."""
        lines = chunk.content.splitlines()
        matched_lines = set()
        for cite in report.citations:
            if cite.source_file == chunk.file_name:
                for l_num in range(cite.start_line, cite.end_line + 1):
                    matched_lines.add(l_num)

        html_snippets = ['<div class="heatmap-box">']
        for idx, line in enumerate(lines, chunk.start_line):
            bg_color = "rgba(74, 222, 128, 0.2)" if idx in matched_lines else "transparent"
            html_snippets.append(
                f'<div style="background-color: {bg_color}; padding: 2px 6px; font-family: monospace;">'
                f'<span style="color: #94a3b8; width: 40px; display: inline-block;">L{idx:03d}:</span> {line}'
                f'</div>'
            )
        html_snippets.append('</div>')
        return "\n".join(html_snippets)
