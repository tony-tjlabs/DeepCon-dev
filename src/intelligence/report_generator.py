"""
PDF 리포트 생성기 — Report Generator v6 (3단 구조 리포트)
=========================================================
Premium consulting-grade PDF report for SK Ecoplant / SK Hynix C-level executives.

v6 업데이트:
  - 3단 구조 적용: 데이터(What) -> 맥락(Why) -> 인사이트(So What)
  - 각 차트 아래 맥락 해석 텍스트 삽입
  - 인사이트 강조 박스 (위험=빨간 배경, 정상=초록 배경)
  - ReportContext 기반 자동 맥락 생성
  - 개선된 여백, 줄간격, 색상 통일

Design philosophy:
  - McKinsey/BCG inspired clean layout with generous whitespace
  - Bold color coding: RED for danger, GREEN for safe, BLUE for primary
  - KPI cards with colored status indicators
  - Professional typography hierarchy (section > sub-header > body > caption)
  - "Page X of Y" footer with TJLABS + SK ecoplant branding
  - No orphaned section titles (title + content always together)
  - Compact charts that stay with their titles

Report flow:
  1. Cover — premium branded hero + KPI summary cards
  2. KPI Summary table + BLE Coverage + Weather
  3. Trend charts (daily workers + EWI/CRE) + 맥락 해석
  4. Risk Analysis — CRE histogram + fatigue scatter + Top 10 + 인사이트 박스
  5. Company Analysis (detailed only) + 맥락 해석
  6. Data-driven Insights (severity cards)
  7. AI Briefing narrative
  8. Daily Detail table (detailed only)
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.intelligence.models import InsightReport, Severity
from src.utils.anonymizer import mask_name
# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH

logger = logging.getLogger(__name__)

# ─── Color Palette (Executive) ────────────────────────────────────────
_C = {
    "primary":    (17, 55, 108),      # #11376C — SK deep navy
    "accent":     (0, 120, 212),      # #0078D4 — bright blue
    "text":       (34, 34, 34),       # #222222 — near-black for readability
    "text_dark":  (17, 17, 17),       # #111111 — headings
    "muted":      (110, 120, 135),    # #6E7887 — secondary text
    "caption":    (130, 140, 155),    # #828C9B — captions
    "bg_section": (245, 247, 250),    # #F5F7FA — section background
    "bg_kpi":     (234, 242, 252),    # #EAF2FC — KPI card background
    "white":      (255, 255, 255),
    "danger":     (200, 40, 35),      # #C82823 — bold red
    "danger_bg":  (255, 235, 235),    # #FFEBEB — danger card bg
    "warning":    (220, 120, 20),     # #DC7814 — warning orange
    "warning_bg": (255, 243, 224),    # #FFF3E0 — warning card bg
    "success":    (30, 150, 75),      # #1E964B — success green
    "success_bg": (232, 248, 237),    # #E8F8ED — success card bg
    "row_even":   (255, 255, 255),
    "row_odd":    (248, 250, 253),    # #F8FAFD — subtle stripe
    "header_bg":  (17, 55, 108),      # deep navy
    "header_fg":  (255, 255, 255),
    "cover_hero": (17, 55, 108),
    "cover_bar":  (0, 120, 212),
    "footer":     (140, 150, 165),
    "divider":    (210, 218, 228),    # #D2DAE4
    "gold":       (180, 145, 50),     # #B49132 — premium accent
}

# ─── Font ─────────────────────────────────────────────────────────────
_BUNDLED_FONT = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts" / "NotoSansKR.ttf"
_FONT_PATHS = [
    str(_BUNDLED_FONT),
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKkr-Regular.otf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def _find_korean_font() -> str | None:
    for p in _FONT_PATHS:
        if Path(p).exists():
            return p
    return None


# ─── PDF Wrapper ──────────────────────────────────────────────────────

class _DeepConPDF:
    """Thin wrapper around fpdf2 with font, footer, and page-break helpers."""

    def __init__(self, report_type: str = "executive"):
        from fpdf import FPDF
        self._pdf = FPDF()
        self._pdf.set_margins(18, 15, 18)
        self._pdf.set_auto_page_break(auto=True, margin=18)
        self.report_type = report_type
        self._page_count = 0
        self._total_pages_placeholder = "{nb}"
        self._pdf.alias_nb_pages(self._total_pages_placeholder)
        self._setup_font()

    def _setup_font(self):
        font_path = _find_korean_font()
        if font_path:
            try:
                self._pdf.add_font("korean", "", font_path, uni=True)
                self._pdf.add_font("korean", "B", font_path, uni=True)
                self._font = "korean"
            except Exception as e:
                logger.warning(f"Korean font load failed: {e}")
                self._font = "Helvetica"
        else:
            self._font = "Helvetica"

    def add_page(self, skip_footer: bool = False):
        self._pdf.add_page()
        self._page_count += 1
        if not skip_footer:
            self._write_footer()

    def _write_footer(self):
        self._pdf.set_auto_page_break(auto=False)
        self._pdf.set_y(-14)
        # Thin line
        self._pdf.set_draw_color(*_C["divider"])
        self._pdf.set_line_width(0.2)
        self._pdf.line(18, self._pdf.get_y(), 192, self._pdf.get_y())
        self._pdf.set_y(-11)
        self._pdf.set_font(self._font, "", 6.5)
        self._pdf.set_text_color(*_C["footer"])
        self._pdf.cell(
            100, 3,
            "Designed & Developed by TJLABS  |  In partnership with SK ecoplant  |  Confidential",
            align="L",
        )
        self._pdf.cell(
            0, 3,
            f"Page {self._page_count} of {self._total_pages_placeholder}",
            align="R",
        )
        self._pdf.set_auto_page_break(auto=True, margin=18)
        self._pdf.set_y(17)

    def ensure_space(self, mm: float = 60):
        """Add new page if less than `mm` remaining before bottom margin."""
        remaining = self._pdf.h - self._pdf.get_y() - 18
        if remaining < mm:
            self.add_page()

    def output(self) -> bytes:
        result = self._pdf.output()
        return bytes(result) if isinstance(result, bytearray) else result

    @property
    def pdf(self):
        return self._pdf

    @property
    def font(self):
        return self._font

    @property
    def content_width(self) -> float:
        return self._pdf.w - self._pdf.l_margin - self._pdf.r_margin


# ─── Typographic helpers ─────────────────────────────────────────────

def _safe_multi_cell(pdf, h: float, text: str, **kwargs):
    """Reset X to left margin then multi_cell (auto_page_break defense)."""
    pdf.set_x(pdf.l_margin)
    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(w, h, text, **kwargs)


def _section_title(report: _DeepConPDF, title: str, min_space: float = 55):
    """Section title with accent underline. Ensures title + content stay together."""
    report.ensure_space(min_space)
    pdf = report.pdf
    pdf.ln(8)
    pdf.set_x(pdf.l_margin)
    pdf.set_font(report.font, "B", 13)
    pdf.set_text_color(*_C["primary"])
    pdf.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    # Thick accent bar
    pdf.set_fill_color(*_C["accent"])
    pdf.rect(pdf.l_margin, y, 45, 1.2, "F")
    pdf.ln(5)


def _sub_header(pdf, font: str, title: str):
    """10pt bold sub-header with left accent dot."""
    pdf.ln(5)
    pdf.set_x(pdf.l_margin)
    # Small accent dot
    pdf.set_fill_color(*_C["accent"])
    y_dot = pdf.get_y() + 2.5
    pdf.rect(pdf.l_margin, y_dot, 2.5, 2.5, "F")
    pdf.set_x(pdf.l_margin + 5)
    pdf.set_font(font, "B", 10)
    pdf.set_text_color(*_C["text_dark"])
    pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _body_text(pdf, font: str, text: str):
    """9pt body text in near-black."""
    pdf.set_font(font, "", 9)
    pdf.set_text_color(*_C["text"])
    _safe_multi_cell(pdf, 4.5, text)
    pdf.ln(1)


def _analysis_comment(pdf, font: str, text: str):
    """Analysis text below charts — dark text for readability."""
    pdf.set_font(font, "", 8.5)
    pdf.set_text_color(*_C["text"])
    # Light background box
    y_start = pdf.get_y()
    pdf.set_fill_color(*_C["bg_section"])
    pdf.set_x(pdf.l_margin)
    # Render text with background
    _safe_multi_cell(pdf, 4.5, text, fill=True)
    pdf.ln(3)


def _context_label(pdf, font: str, data_text: str, context_text: str):
    """
    맥락 레이블 (v6 신규): 차트 아래 데이터 요약 + 맥락 해석.

    [데이터]: 수치 요약 (진한 텍스트)
    [맥락]: 의미 해석 (연한 텍스트)
    """
    # 데이터 요약
    if data_text:
        pdf.set_font(font, "B", 8)
        pdf.set_text_color(*_C["accent"])
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 4.5, "[데이터] ", new_x="RIGHT")
        pdf.set_font(font, "", 8)
        pdf.set_text_color(*_C["text"])
        _safe_multi_cell(pdf, 4.5, data_text)

    # 맥락 해석
    if context_text:
        pdf.set_fill_color(*_C["bg_section"])
        pdf.set_font(font, "", 8)
        pdf.set_text_color(*_C["muted"])
        pdf.set_x(pdf.l_margin)
        _safe_multi_cell(pdf, 4.5, f"[맥락] {context_text}", fill=True)

    pdf.ln(2)


def _insight_box(pdf, font: str, text: str, severity: str = "normal"):
    """
    인사이트 강조 박스 (v6 신규): 섹션 끝 핵심 메시지.

    severity: "danger" | "warning" | "success" | "normal"
    """
    if not text:
        return

    # 색상 매핑
    color_map = {
        "danger": (_C["danger_bg"], _C["danger"]),
        "warning": (_C["warning_bg"], _C["warning"]),
        "success": (_C["success_bg"], _C["success"]),
        "normal": (_C["bg_section"], _C["accent"]),
    }
    bg_color, border_color = color_map.get(severity, color_map["normal"])
    label_map = {
        "danger": "주목 필요",
        "warning": "주의",
        "success": "양호",
        "normal": "참고",
    }
    label = label_map.get(severity, "참고")

    # 박스 렌더링
    y_start = pdf.get_y()
    box_h = 12
    pdf.set_fill_color(*bg_color)
    pdf.rect(pdf.l_margin, y_start, pdf.w - pdf.l_margin - pdf.r_margin, box_h, "F")
    # 좌측 두꺼운 테두리
    pdf.set_fill_color(*border_color)
    pdf.rect(pdf.l_margin, y_start, 4, box_h, "F")

    # 라벨
    pdf.set_font(font, "B", 8)
    pdf.set_text_color(*border_color)
    pdf.set_xy(pdf.l_margin + 7, y_start + 2)
    pdf.cell(30, 4, f"[{label}]")

    # 텍스트
    pdf.set_font(font, "", 8.5)
    pdf.set_text_color(*_C["text"])
    pdf.set_x(pdf.l_margin + 7)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 10, 4.5, text)

    pdf.set_y(y_start + box_h + 3)
    pdf.ln(2)


def _caption_text(pdf, font: str, text: str):
    """Small caption text for definitions and footnotes."""
    pdf.set_font(font, "", 7)
    pdf.set_text_color(*_C["caption"])
    _safe_multi_cell(pdf, 3.5, text)
    pdf.set_text_color(*_C["text"])
    pdf.ln(2)


def _embed_chart(report: _DeepConPDF, chart_bytes: bytes, w: float = 140):
    """Insert PNG bytes — centered on page."""
    if not chart_bytes:
        return
    report.ensure_space(55)
    pdf = report.pdf
    x = (210 - w) / 2
    pdf.image(io.BytesIO(chart_bytes), x=x, w=w)
    pdf.ln(2)


def _draw_table(
    report: _DeepConPDF,
    headers: list[str],
    widths: list[float],
    rows: list[list[str]],
    aligns: list[str] | None = None,
    row_height: float = 6.5,
    highlight_col: int | None = None,
    highlight_threshold: float | None = None,
    highlight_color: tuple = _C["danger"],
    success_threshold: float | None = None,
    success_color: tuple = _C["success"],
):
    """Professional table with navy header and alternating rows."""
    pdf = report.pdf
    if aligns is None:
        aligns = ["L"] + ["R"] * (len(headers) - 1)

    total_w = sum(widths)

    # Header
    pdf.set_x(pdf.l_margin)
    pdf.set_font(report.font, "B", 7.5)
    pdf.set_fill_color(*_C["header_bg"])
    pdf.set_text_color(*_C["header_fg"])
    for h, w, a in zip(headers, widths, aligns):
        pdf.cell(w, 7, f" {h} ", border=0, fill=True, align="C")
    pdf.ln()

    # Rows
    pdf.set_font(report.font, "", 7.5)
    for i, row_data in enumerate(rows):
        # Skip empty separator rows
        if all(v.strip() == "" for v in row_data):
            continue

        bg = _C["row_even"] if i % 2 == 0 else _C["row_odd"]
        pdf.set_fill_color(*bg)
        pdf.set_x(pdf.l_margin)

        for j, (val, w, a) in enumerate(zip(row_data, widths, aligns)):
            # Color coding for highlight column
            if highlight_col is not None and j == highlight_col:
                try:
                    fval = float(val)
                    if highlight_threshold is not None and fval >= highlight_threshold:
                        pdf.set_text_color(*highlight_color)
                        pdf.set_font(report.font, "B", 7.5)
                    elif success_threshold is not None and fval < success_threshold:
                        pdf.set_text_color(*success_color)
                        pdf.set_font(report.font, "", 7.5)
                    else:
                        pdf.set_text_color(*_C["text"])
                        pdf.set_font(report.font, "", 7.5)
                except (ValueError, TypeError):
                    pdf.set_text_color(*_C["text"])
                    pdf.set_font(report.font, "", 7.5)
            else:
                pdf.set_text_color(*_C["text"])
                pdf.set_font(report.font, "", 7.5)

            cell_text = f" {val} " if a == "L" else f"{val} "
            pdf.cell(w, row_height, cell_text, border=0, fill=True, align=a)
        pdf.ln()

    # Bottom line
    pdf.set_draw_color(*_C["divider"])
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + total_w, pdf.get_y())
    pdf.ln(3)


def _draw_kpi_card(
    pdf, font: str, x: float, y: float, w: float, h: float,
    label: str, value: str, sub: str = "",
    color: tuple = _C["accent"], bg: tuple = _C["bg_kpi"],
):
    """Draw a single KPI card with colored top accent."""
    # Background
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(*_C["divider"])
    pdf.set_line_width(0.3)
    pdf.rect(x, y, w, h, "DF")
    # Top color accent bar
    pdf.set_fill_color(*color)
    pdf.rect(x, y, w, 2, "F")
    # Label
    pdf.set_xy(x, y + 4)
    pdf.set_font(font, "", 7)
    pdf.set_text_color(*_C["muted"])
    pdf.cell(w, 4, label, align="C")
    # Value
    pdf.set_xy(x, y + 10)
    pdf.set_font(font, "B", 14)
    pdf.set_text_color(*color)
    pdf.cell(w, 8, value, align="C")
    # Sub
    if sub:
        pdf.set_xy(x, y + 21)
        pdf.set_font(font, "", 7)
        pdf.set_text_color(*_C["muted"])
        pdf.cell(w, 4, sub, align="C")


def _value_color(value: float, thresholds: tuple = (0.4, 0.6)) -> tuple:
    """Return color based on value thresholds (for CRE-type metrics)."""
    if value >= thresholds[1]:
        return _C["danger"]
    elif value >= thresholds[0]:
        return _C["warning"]
    return _C["success"]


# ─── Main entry point ────────────────────────────────────────────────

def generate_report(
    report_type: str,
    date_range: str,
    sector_label: str,
    insights: InsightReport | None,
    kpi_data: dict,
    llm_narrative: str = "",
    worker_stats: dict | None = None,
    company_data: pd.DataFrame | None = None,
    trend_data: list[dict] | None = None,
    worker_df: pd.DataFrame | None = None,
    metas: list[dict] | None = None,
    dates: list[str] | None = None,
    use_llm_context: bool = False,
) -> bytes:
    """
    Generate premium PDF report (v6).

    Args:
        report_type: "executive" | "detailed"
        date_range: 날짜 범위 문자열
        sector_label: 현장 라벨
        insights: InsightReport 객체
        kpi_data: KPI 딕셔너리
        llm_narrative: LLM 생성 내러티브
        worker_stats: 작업자 통계
        company_data: 업체별 DataFrame
        trend_data: 트렌드 데이터 리스트
        worker_df: 작업자별 DataFrame
        metas: 메타 데이터 리스트
        dates: 날짜 목록
        use_llm_context: LLM 맥락 생성 사용 여부

    Returns:
        PDF 바이트
    """
    report = _DeepConPDF(report_type)

    # v6: ReportContext 생성
    report_ctx = None
    try:
        from src.intelligence.report_context import build_report_context
        report_ctx = build_report_context(
            worker_df=worker_df,
            kpi=kpi_data,
            dates=dates,
            use_llm=use_llm_context,
        )
    except Exception as e:
        logger.warning(f"ReportContext 생성 실패: {e}")

    # 1. Cover
    _add_cover(report, date_range, sector_label, kpi_data)

    # 2. KPI + BLE Coverage
    _add_kpi_page(report, kpi_data, worker_stats, worker_df)

    # 3. Trend Charts + 맥락
    if metas and dates and len(dates) >= 2:
        _add_trend_page(report, kpi_data, worker_df, metas, dates, report_ctx)

    # 4. Risk Analysis + 맥락 + 인사이트 박스
    _add_risk_analysis_page(report, worker_df, kpi_data, worker_stats, report_ctx)

    # 5. Company Analysis (detailed only) + 맥락
    if report_type == "detailed" and worker_df is not None:
        _add_company_analysis_page(report, worker_df, company_data, report_ctx)

    # 6. AI Insights
    if insights and insights.insights:
        _add_insights_page(report, insights)

    # 7. AI Briefing
    if llm_narrative:
        _add_narrative_page(report, llm_narrative)

    # 8. Daily Detail table (detailed only)
    if report_type == "detailed" and trend_data:
        _add_trend_table_page(report, trend_data)

    return report.output()


# ─── Page builders ───────────────────────────────────────────────────

def _add_cover(report: _DeepConPDF, date_range: str, sector_label: str, kpi: dict):
    """Premium branded cover page."""
    pdf = report.pdf
    report.add_page(skip_footer=True)

    # Full-width navy header bar
    pdf.set_fill_color(*_C["cover_hero"])
    pdf.rect(0, 0, 210, 70, "F")

    # Accent stripe
    pdf.set_fill_color(*_C["cover_bar"])
    pdf.rect(0, 70, 210, 3, "F")

    # DeepCon logo on dark background
    pdf.set_y(22)
    pdf.set_font(report.font, "B", 42)
    pdf.set_text_color(*_C["white"])
    pdf.cell(0, 18, "DeepCon", align="C", new_x="LMARGIN", new_y="NEXT")

    # Tagline
    pdf.set_font(report.font, "", 10)
    pdf.set_text_color(180, 200, 230)
    pdf.cell(0, 6, "Spatial AI  |  Construction Safety & Productivity Platform",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # Report type card below the hero
    pdf.set_y(85)
    type_label = "Executive Summary" if report.report_type == "executive" else "Detailed Report"
    pdf.set_font(report.font, "B", 18)
    pdf.set_text_color(*_C["text_dark"])
    pdf.cell(0, 12, type_label, align="C", new_x="LMARGIN", new_y="NEXT")

    # Sector
    pdf.set_font(report.font, "", 12)
    pdf.set_text_color(*_C["muted"])
    pdf.cell(0, 8, sector_label, align="C", new_x="LMARGIN", new_y="NEXT")

    # Date range
    pdf.ln(2)
    pdf.set_font(report.font, "B", 11)
    pdf.set_text_color(*_C["accent"])
    pdf.cell(0, 8, date_range, align="C", new_x="LMARGIN", new_y="NEXT")

    # ── KPI Cards Row 1: Personnel ──
    pdf.ln(8)
    cum_a = kpi.get("cum_access", kpi.get("total_access", 0))
    avg_a = kpi.get("avg_daily_access", cum_a)
    unique_w = kpi.get("unique_workers", 0)
    tward_rate = kpi.get("tward_rate", 0)

    card_w = 54
    gap = 4
    start_x = (210 - (card_w * 3 + gap * 2)) / 2
    card_y = pdf.get_y()

    _draw_kpi_card(pdf, report.font, start_x, card_y, card_w, 32,
                   "Total Access (Period)", f"{cum_a:,}",
                   f"Daily avg {avg_a:,}",
                   color=_C["primary"])
    _draw_kpi_card(pdf, report.font, start_x + card_w + gap, card_y, card_w, 32,
                   "Unique Workers", f"{unique_w:,}",
                   f"{kpi.get('companies', 0)} Companies",
                   color=_C["accent"])
    _draw_kpi_card(pdf, report.font, start_x + 2 * (card_w + gap), card_y, card_w, 32,
                   "T-Ward Rate", f"{tward_rate:.1f}%",
                   "BLE Tracking Coverage",
                   color=_C["success"] if tward_rate >= 70 else _C["warning"] if tward_rate >= 50 else _C["danger"])

    pdf.set_y(card_y + 38)

    # ── KPI Cards Row 2: Risk Metrics ──
    avg_ewi = kpi.get("avg_ewi", 0)
    avg_cre = kpi.get("avg_cre", 0)
    high_cre = kpi.get("high_cre", 0)

    card_y2 = pdf.get_y()
    _draw_kpi_card(pdf, report.font, start_x, card_y2, card_w, 32,
                   "Avg EWI (Productivity)", f"{avg_ewi:.3f}",
                   "Effective Work Intensity",
                   color=_C["accent"])
    _draw_kpi_card(pdf, report.font, start_x + card_w + gap, card_y2, card_w, 32,
                   "Avg CRE (Risk)", f"{avg_cre:.3f}",
                   "Construction Risk Exposure",
                   color=_value_color(avg_cre))
    _draw_kpi_card(pdf, report.font, start_x + 2 * (card_w + gap), card_y2, card_w, 32,
                   f"High Risk (CRE>={CRE_HIGH:.1f})", f"{high_cre:,}",
                   "Workers requiring attention",
                   color=_C["danger"] if high_cre > 0 else _C["success"])

    # ── Cover bottom section: summary line + footer ──
    pdf.set_auto_page_break(auto=False)

    # Summary insight line above footer
    pdf.set_y(-50)
    pdf.set_fill_color(*_C["bg_section"])
    pdf.rect(18, pdf.get_y(), 174, 20, "F")
    pdf.set_xy(22, pdf.get_y() + 3)
    pdf.set_font(report.font, "", 8)
    pdf.set_text_color(*_C["text"])

    # Quick summary line
    summary_parts = []
    if cum_a > 0:
        summary_parts.append(f"Period total: {cum_a:,} entries")
    if high_cre > 0:
        summary_parts.append(f"High-risk workers: {high_cre:,}")
    if tward_rate > 0:
        summary_parts.append(f"T-Ward compliance: {tward_rate:.1f}%")
    summary_line = "  |  ".join(summary_parts) if summary_parts else ""
    if summary_line:
        pdf.cell(170, 5, summary_line, align="C")
    pdf.set_xy(22, pdf.get_y() + 8)
    pdf.set_font(report.font, "", 7)
    pdf.set_text_color(*_C["caption"])
    pdf.cell(170, 4, "This report contains proprietary data. Distribution is restricted to authorized personnel only.", align="C")

    # Footer
    pdf.set_y(-24)

    # Gold accent line
    pdf.set_draw_color(*_C["gold"])
    pdf.set_line_width(0.5)
    pdf.line(70, pdf.get_y(), 140, pdf.get_y())

    pdf.set_y(-20)
    pdf.set_font(report.font, "", 8)
    pdf.set_text_color(*_C["muted"])
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(report.font, "", 7.5)
    pdf.set_text_color(*_C["footer"])
    pdf.cell(0, 4, "Designed & Developed by TJLABS  |  In partnership with SK ecoplant  |  CONFIDENTIAL",
             align="C")
    pdf.set_auto_page_break(auto=True, margin=18)


def _add_kpi_page(
    report: _DeepConPDF, kpi: dict, worker_stats: dict | None,
    worker_df: pd.DataFrame | None,
):
    """KPI table + BLE coverage + Weather."""
    pdf = report.pdf
    report.add_page()

    _section_title(report, "Key Performance Indicators", min_space=80)

    # ── KPI table ──
    cum_access = kpi.get("cum_access", kpi.get("total_access", 0))
    avg_daily = kpi.get("avg_daily_access", cum_access)
    unique_w = kpi.get("unique_workers", 0)
    cum_tward = kpi.get("cum_tward", kpi.get("total_tward", 0))

    kpi_rows = [
        ["Total Access (Period)", f"{cum_access:,}",
         "Total entries during the reporting period"],
        ["Daily Average Access", f"{avg_daily:,}",
         "Average daily entries via biometric system"],
        ["Unique Workers", f"{unique_w:,}" if unique_w else "--",
         "Distinct worker IDs over the period"],
        ["T-Ward Equipped", f"{cum_tward:,} ({kpi.get('tward_rate', 0):.1f}%)",
         "Workers with BLE tracking device"],
    ]

    # Add weekday/weekend if available
    wd_days = kpi.get("weekday_days", 0)
    we_days = kpi.get("weekend_days", 0)
    if wd_days > 0:
        kpi_rows.append([
            f"  Weekday Avg ({wd_days}d)", f"{kpi.get('weekday_avg_access', 0):,}",
            "Mon-Fri average"])
    if we_days > 0:
        kpi_rows.append([
            f"  Weekend Avg ({we_days}d)", f"{kpi.get('weekend_avg_access', 0):,}",
            "Sat-Sun average"])

    # EWI/CRE
    kpi_rows.append(["Avg EWI", f"{kpi.get('avg_ewi', 0):.3f}",
                      "Effective Work Intensity (0-1)"])
    kpi_rows.append(["Avg CRE", f"{kpi.get('avg_cre', 0):.3f}",
                      "Construction Risk Exposure (0-1)"])
    kpi_rows.append([f"High Risk (CRE >= {CRE_HIGH:.1f})", f"{kpi.get('high_cre', 0):,}",
                      "Workers above danger threshold"])

    if worker_stats:
        kpi_rows.append(["High Fatigue (>= 0.6)", f"{worker_stats.get('high_fatigue', 0):,}",
                          "Workers with high fatigue score"])
        kpi_rows.append(["Confined Space Entry", f"{worker_stats.get('confined', 0):,}",
                          "Workers who entered confined spaces"])

    _draw_table(
        report,
        headers=["Metric", "Value", "Description"],
        widths=[55, 40, 79],
        rows=kpi_rows,
        aligns=["L", "R", "L"],
        row_height=6.5,
    )

    # Metric definitions
    _caption_text(pdf, report.font,
        "EWI (Effective Work Intensity): Ratio of active work time to total shift duration. "
        "CRE (Construction Risk Exposure): Composite index of spatial risk, fatigue, and solo-work factors. "
        "Unique ID: Distinct workers who entered at least once during the period."
    )

    # ── BLE Coverage ──
    if worker_df is not None and "ble_coverage" in worker_df.columns:
        _sub_header(pdf, report.font, "Data Quality (BLE Coverage)")
        cov = worker_df["ble_coverage"].value_counts()
        total_w = len(worker_df)
        normal = cov.get("\uc815\uc0c1", 0)
        partial = cov.get("\ubd80\ubd84\uc74c\uc601", 0)
        shadow = cov.get("\uc74c\uc601", 0)
        unmeas = cov.get("\ubbf8\uce21\uc815", 0)
        reliable_pct = round(normal / total_w * 100, 1) if total_w else 0

        ble_rows = [
            ["Normal (80%+)", f"{normal:,} ({reliable_pct}%)", "Reliable tracking data"],
            ["Partial (50-80%)", f"{partial:,}", "Intermittent signal"],
            ["Shadow (20-50%)", f"{shadow:,}", "Weak signal coverage"],
            ["Unmeasured (<20%)", f"{unmeas:,}", "No reliable BLE data"],
        ]
        _draw_table(
            report,
            headers=["Coverage Level", "Workers", "Status"],
            widths=[55, 40, 79],
            rows=ble_rows,
            aligns=["L", "R", "L"],
            row_height=6.5,
        )

        if unmeas > 0:
            _analysis_comment(pdf, report.font,
                f"Note: {unmeas:,} workers have insufficient BLE data. "
                f"Possible causes: T-Ward not worn, sensor shadow zones, or device malfunction. "
                f"EWI/CRE metrics for these workers are unreliable.")

    # ── Weather summary ──
    weather_ewi = kpi.get("weather_ewi")
    if weather_ewi and isinstance(weather_ewi, dict):
        _sub_header(pdf, report.font, "Weather Impact on Metrics")
        w_rows = []
        for tag, label in [("sunny", "Sunny"), ("rain", "Rain"), ("snow", "Snow")]:
            days = weather_ewi.get(f"{tag}_days", 0)
            if days > 0:
                avg_e = weather_ewi.get(f"{tag}_avg_ewi", 0)
                avg_c = weather_ewi.get(f"{tag}_avg_cre", 0)
                w_rows.append([label, str(days), f"{avg_e:.3f}", f"{avg_c:.3f}"])
        if w_rows:
            _draw_table(
                report,
                headers=["Weather", "Days", "Avg EWI", "Avg CRE"],
                widths=[50, 30, 47, 47],
                rows=w_rows,
                aligns=["L", "R", "R", "R"],
                row_height=6.5,
            )


def _add_trend_page(
    report: _DeepConPDF, kpi: dict,
    worker_df: pd.DataFrame | None,
    metas: list[dict], dates: list[str],
    report_ctx=None,
):
    """Trend charts page — daily workers + EWI/CRE + 맥락 해석 (v6)."""
    pdf = report.pdf
    # Start new page for trends (these charts need full page)
    report.add_page()

    _section_title(report, "Trend Analysis", min_space=70)

    weather_df_for_chart = kpi.get("_weather_df")

    try:
        from src.intelligence.report_charts import chart_daily_trend, chart_ewi_cre_trend

        _sub_header(pdf, report.font, "Daily Access Trend")
        trend_img = chart_daily_trend(metas, dates, weather_df=weather_df_for_chart)
        _embed_chart(report, trend_img, w=150)

        # v6: 일별 출입인원 맥락 레이블
        if metas:
            access_vals = [m.get("total_workers_access", 0) for m in metas]
            tward_vals = [m.get("total_workers_move", 0) for m in metas]
            avg_access = sum(access_vals) / len(access_vals) if access_vals else 0
            avg_tward = sum(tward_vals) / len(tward_vals) if tward_vals else 0
            tward_rate = avg_tward / avg_access * 100 if avg_access > 0 else 0
            _context_label(
                pdf, report.font,
                f"일평균 출입 {avg_access:,.0f}명, T-Ward {avg_tward:,.0f}명 ({tward_rate:.1f}%)",
                "T-Ward 착용률은 BLE 추적 커버리지를 나타냅니다. 70% 이상이 권장됩니다.",
            )

        if worker_df is not None and "date" in worker_df.columns:
            _sub_header(pdf, report.font, "Daily EWI / CRE Trend")
            ewi_cre_img = chart_ewi_cre_trend(worker_df, dates, weather_df=weather_df_for_chart)
            _embed_chart(report, ewi_cre_img, w=150)

            # v6: EWI/CRE 트렌드 맥락 레이블
            if report_ctx and report_ctx.ewi_cre_trend.data_comment:
                _context_label(
                    pdf, report.font,
                    report_ctx.ewi_cre_trend.data_comment,
                    report_ctx.ewi_cre_trend.context_text,
                )
                # 인사이트 박스
                if report_ctx.ewi_cre_trend.insight_text:
                    _insight_box(
                        pdf, report.font,
                        report_ctx.ewi_cre_trend.insight_text,
                        report_ctx.ewi_cre_trend.severity,
                    )

            # Trend analysis text
            daily_agg = (
                worker_df.groupby("date")
                .agg(avg_ewi=("ewi", "mean"), avg_cre=("cre", "mean"))
                .reset_index().sort_values("date")
            )
            if len(daily_agg) >= 2:
                first, last = daily_agg.iloc[0], daily_agg.iloc[-1]
                ewi_d = last["avg_ewi"] - first["avg_ewi"]
                cre_d = last["avg_cre"] - first["avg_cre"]
                ewi_dir = "UP" if ewi_d > 0.01 else "DOWN" if ewi_d < -0.01 else "STABLE"
                cre_dir = "UP" if cre_d > 0.01 else "DOWN" if cre_d < -0.01 else "STABLE"
                _analysis_comment(pdf, report.font,
                    f"EWI: {first['avg_ewi']:.3f} -> {last['avg_ewi']:.3f} ({ewi_dir})  |  "
                    f"CRE: {first['avg_cre']:.3f} -> {last['avg_cre']:.3f} ({cre_dir}).  "
                    f"Overall trend: EWI {ewi_dir}, CRE {cre_dir}.")
    except Exception as e:
        logger.warning(f"Trend chart failed: {e}")


def _add_risk_analysis_page(
    report: _DeepConPDF,
    worker_df: pd.DataFrame | None,
    kpi: dict,
    worker_stats: dict | None,
    report_ctx=None,
):
    """Risk Analysis — CRE distribution + fatigue x CRE + Top 10 + 맥락/인사이트 (v6)."""
    pdf = report.pdf
    # Don't force new page - let it flow from previous content
    _section_title(report, "Risk Analysis", min_space=80)

    if worker_df is None or worker_df.empty:
        pdf.set_font(report.font, "", 10)
        pdf.set_text_color(*_C["muted"])
        pdf.cell(0, 8, "No worker data available.")
        return

    # ── CRE Distribution ──
    if "cre" in worker_df.columns:
        try:
            from src.intelligence.report_charts import chart_cre_distribution
            _sub_header(pdf, report.font, "CRE (Risk Exposure) Distribution")
            cre_img = chart_cre_distribution(worker_df)
            _embed_chart(report, cre_img, w=150)

            # v6: CRE 분포 맥락 레이블
            if report_ctx and report_ctx.cre_distribution.data_comment:
                _context_label(
                    pdf, report.font,
                    report_ctx.cre_distribution.data_comment,
                    report_ctx.cre_distribution.context_text,
                )
            else:
                # 폴백: 기존 코멘트 — CRE_HIGH 기준
                high = int((worker_df["cre"].fillna(0) >= CRE_HIGH).sum())
                med = int(((worker_df["cre"] >= 0.3) & (worker_df["cre"] < CRE_HIGH)).sum())
                low = int((worker_df["cre"] < 0.3).sum())
                total = len(worker_df) or 1
                _analysis_comment(pdf, report.font,
                    f"Low Risk (<0.3): {low:,} ({low/total*100:.1f}%)  |  "
                    f"Medium (0.3-{CRE_HIGH:.1f}): {med:,} ({med/total*100:.1f}%)  |  "
                    f"High Risk (>={CRE_HIGH:.1f}): {high:,} ({high/total*100:.1f}%).  "
                    f"{high:,} workers require immediate safety monitoring.")
        except Exception as e:
            logger.warning(f"CRE chart failed: {e}")

    # ── Fatigue x CRE scatter ──
    if "fatigue_score" in worker_df.columns and "cre" in worker_df.columns:
        try:
            from src.intelligence.report_charts import chart_fatigue_vs_cre
            report.ensure_space(65)
            _sub_header(pdf, report.font, "Fatigue x CRE Compound Risk")
            fatigue_img = chart_fatigue_vs_cre(worker_df)
            _embed_chart(report, fatigue_img, w=140)

            # v6: 피로 x CRE 맥락 레이블
            if report_ctx and report_ctx.fatigue_vs_cre.data_comment:
                _context_label(
                    pdf, report.font,
                    report_ctx.fatigue_vs_cre.data_comment,
                    report_ctx.fatigue_vs_cre.context_text,
                )
                # v6: 인사이트 박스 (복합 고위험)
                if report_ctx.fatigue_vs_cre.insight_text:
                    _insight_box(
                        pdf, report.font,
                        report_ctx.fatigue_vs_cre.insight_text,
                        report_ctx.fatigue_vs_cre.severity,
                    )
            else:
                both = int(((worker_df["fatigue_score"].fillna(0) >= 0.6) & (worker_df["cre"].fillna(0) >= CRE_HIGH)).sum())
                _analysis_comment(pdf, report.font,
                    f"Compound high-risk workers (Fatigue >= 0.6 AND CRE >= {CRE_HIGH:.1f}): {both:,}.  "
                    f"These workers face significantly elevated accident probability and "
                    f"require immediate rest rotation or reassignment to lower-risk zones.")
        except Exception as e:
            logger.warning(f"Fatigue chart failed: {e}")

    # ── Top 10 High-Risk Workers ──
    if "cre" in worker_df.columns and "user_name" in worker_df.columns:
        # Need space for: sub_header(~14) + header row(7) + 10 data rows(65) + bottom line(3) = ~89
        report.ensure_space(100)
        _sub_header(pdf, report.font, "Top 10 Highest-Risk Workers")
        top10 = worker_df.nlargest(10, "cre")

        top10_rows = []
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            top10_rows.append([
                str(rank),
                mask_name(row.get("user_name", ""))[:12],
                str(row.get("company_name", ""))[:18],
                f"{row.get('cre', 0):.3f}",
                f"{row.get('ewi', 0):.3f}" if "ewi" in row else "--",
                f"{row.get('fatigue_score', 0):.2f}" if "fatigue_score" in row else "--",
            ])

        _draw_table(
            report,
            headers=["#", "Worker", "Company", "CRE", "EWI", "Fatigue"],
            widths=[12, 30, 48, 26, 26, 26],
            rows=top10_rows,
            aligns=["C", "L", "L", "R", "R", "R"],
            row_height=6.5,
            highlight_col=3,
            highlight_threshold=0.6,
        )


def _add_company_analysis_page(
    report: _DeepConPDF,
    worker_df: pd.DataFrame,
    company_data: pd.DataFrame | None,
    report_ctx=None,
):
    """Company analysis — chart + table (detailed only) + 맥락 해석 (v6). Filters out unknown."""
    pdf = report.pdf
    report.add_page()

    _section_title(report, "Company Analysis", min_space=80)

    # Company risk chart (excluding unknown)
    if "company_name" in worker_df.columns and "cre" in worker_df.columns:
        try:
            from src.intelligence.report_charts import chart_company_risk
            chart_df = worker_df[worker_df["company_name"] != "\ubbf8\ud655\uc778"]
            if not chart_df.empty:
                _sub_header(pdf, report.font, "Average CRE by Company (Top 10, 5+ workers)")
                comp_img = chart_company_risk(chart_df, top_n=10)
                _embed_chart(report, comp_img, w=150)

                # v6: 업체별 분석 맥락 레이블
                if report_ctx and report_ctx.company_risk.data_comment:
                    _context_label(
                        pdf, report.font,
                        report_ctx.company_risk.data_comment,
                        report_ctx.company_risk.context_text,
                    )
                    # 인사이트 박스
                    if report_ctx.company_risk.insight_text:
                        _insight_box(
                            pdf, report.font,
                            report_ctx.company_risk.insight_text,
                            report_ctx.company_risk.severity,
                        )
        except Exception as e:
            logger.warning(f"Company chart failed: {e}")

    # Company table
    if company_data is not None and not company_data.empty:
        _sub_header(pdf, report.font, "Company Detail (Top 15 by headcount)")

        filtered = company_data.copy()
        unknown_df = pd.DataFrame()
        if "company_name" in filtered.columns:
            unknown_mask = filtered["company_name"] == "\ubbf8\ud655\uc778"
            unknown_df = filtered[unknown_mask]
            filtered = filtered[~unknown_mask]

        if "avg_ewi" in filtered.columns:
            filtered = filtered[filtered["avg_ewi"] > 0]

        if "worker_count" in filtered.columns:
            top = filtered.nlargest(15, "worker_count")
        else:
            top = filtered.head(15)

        comp_rows = []
        for _, row in top.iterrows():
            comp_rows.append([
                str(row.get("company_name", ""))[:22],
                f"{row.get('worker_count', 0)}",
                f"{row.get('avg_ewi', 0):.3f}" if "avg_ewi" in row else "--",
                f"{row.get('avg_cre', 0):.3f}" if "avg_cre" in row else "--",
            ])

        _draw_table(
            report,
            headers=["Company", "Workers", "Avg EWI", "Avg CRE"],
            widths=[72, 28, 37, 37],
            rows=comp_rows,
            aligns=["L", "R", "R", "R"],
            highlight_col=3,
            highlight_threshold=0.5,
            success_threshold=0.3,
        )

        # Note about unknown workers
        if not unknown_df.empty:
            n_unknown = int(unknown_df.get("worker_count", pd.Series([0])).sum())
            if n_unknown > 0:
                _analysis_comment(pdf, report.font,
                    f"Note: {n_unknown:,} unidentified workers detected in AccessLog "
                    f"without matching TWardData records. "
                    f"Possible causes: missing check-in records or temporary workers.")


def _add_insights_page(report: _DeepConPDF, insights: InsightReport):
    """Data-driven insight cards with severity indicators."""
    pdf = report.pdf
    report.add_page()

    n_show = 8 if report.report_type == "detailed" else 5
    _section_title(report, f"Data-Driven Alerts ({len(insights.insights)} detected)",
                   min_space=60)

    sev_labels = {4: "CRITICAL", 3: "WARNING", 2: "CAUTION", 1: "INFO"}
    sev_colors = {
        4: _C["danger"],
        3: _C["warning"],
        2: (200, 170, 30),   # dark yellow
        1: _C["accent"],
    }
    sev_bg = {
        4: _C["danger_bg"],
        3: _C["warning_bg"],
        2: (255, 250, 230),
        1: _C["success_bg"],
    }

    for ins in insights.top(n_show):
        report.ensure_space(30)
        sev_label = sev_labels.get(ins.severity, "?")
        _, cat_label = ins.category_label
        bg = sev_bg.get(ins.severity, _C["bg_section"])
        sev_color = sev_colors.get(ins.severity, _C["accent"])

        # Card background
        y_start = pdf.get_y()
        card_h = 8
        pdf.set_fill_color(*bg)
        pdf.rect(pdf.l_margin, y_start, report.content_width, card_h, "F")
        # Left accent bar (thick)
        pdf.set_fill_color(*sev_color)
        pdf.rect(pdf.l_margin, y_start, 3.5, card_h, "F")

        # Severity badge + title
        pdf.set_font(report.font, "B", 9)
        pdf.set_text_color(*sev_color)
        pdf.set_xy(pdf.l_margin + 6, y_start)
        pdf.cell(28, card_h, f"[{sev_label}]")
        pdf.set_text_color(*_C["text_dark"])
        pdf.cell(0, card_h, f"{cat_label}  |  {ins.title}",
                 new_x="LMARGIN", new_y="NEXT")

        # Description
        pdf.set_font(report.font, "", 8.5)
        pdf.set_text_color(*_C["text"])
        pdf.set_x(pdf.l_margin + 6)
        w = report.content_width - 6
        pdf.multi_cell(w, 4.5, ins.description)

        # Recommendation
        if ins.recommendation:
            pdf.set_font(report.font, "", 8.5)
            pdf.set_text_color(*_C["accent"])
            pdf.set_x(pdf.l_margin + 6)
            pdf.multi_cell(w, 4.5, f"Recommendation: {ins.recommendation}")

        pdf.ln(4)


def _add_narrative_page(report: _DeepConPDF, narrative: str):
    """AI briefing narrative — full text, no truncation."""
    pdf = report.pdf
    report.add_page()

    _section_title(report, "AI Site Analysis Briefing", min_space=60)

    # Clean markdown
    clean = narrative
    clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
    clean = re.sub(r'^#+\s*', '', clean, flags=re.MULTILINE)
    clean = clean.replace('---', '').strip()

    # Split into paragraphs and render each with left accent bar
    paragraphs = [p.strip() for p in clean.split('\n') if p.strip()]

    for para in paragraphs:
        # Left accent bar for each paragraph
        y_start = pdf.get_y()
        pdf.set_fill_color(*_C["bg_section"])
        pdf.rect(pdf.l_margin, y_start, report.content_width, 5, "F")
        pdf.set_fill_color(*_C["accent"])
        pdf.rect(pdf.l_margin, y_start, 2.5, 5, "F")

        # Indent text past the accent bar
        pdf.set_font(report.font, "", 9)
        pdf.set_text_color(*_C["text"])
        pdf.set_x(pdf.l_margin + 5)
        w = report.content_width - 5
        pdf.multi_cell(w, 5, para)
        pdf.ln(3)

    pdf.ln(2)


def _add_trend_table_page(report: _DeepConPDF, trend_data: list[dict]):
    """Daily detail table (detailed only)."""
    report.add_page()

    _section_title(report, "Daily Detail Data", min_space=60)

    rows = []
    for row in trend_data:
        d = row.get("date", "")
        date_fmt = f"{d[4:6]}/{d[6:]}" if len(d) == 8 else d
        cre_val = row.get("avg_cre", 0)
        rows.append([
            date_fmt,
            f"{row.get('total_access', 0):,}",
            f"{row.get('total_tward', 0):,}",
            f"{row.get('avg_ewi', 0):.3f}",
            f"{cre_val:.3f}",
            f"{row.get('high_cre', 0)}",
        ])

    _draw_table(
        report,
        headers=["Date", "Access", "T-Ward", "Avg EWI", "Avg CRE", "High Risk"],
        widths=[28, 28, 28, 30, 30, 28],
        rows=rows,
        aligns=["C", "R", "R", "R", "R", "R"],
        row_height=7,
        highlight_col=4,
        highlight_threshold=0.5,
        success_threshold=0.35,
    )
