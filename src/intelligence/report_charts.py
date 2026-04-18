"""
PDF 리포트용 차트 생성 — Report Charts v2
==========================================
Plotly 차트 -> PNG 바이트 변환.
fpdf2의 image()에 직접 삽입 가능한 바이트를 반환.

v2 changes:
  - matplotlib fallback: rotate X-axis labels 45 deg when >10 values
  - bar chart label overlap prevention
  - consistent color scheme matching report_generator v4
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ★ 임계값 단일 소스 — src/metrics/constants.py
from src.metrics.constants import CRE_HIGH

logger = logging.getLogger(__name__)

# ─── Shared constants ────────────────────────────────────────────────
_PRINT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#F8F9FC",
    font=dict(family="Arial, Helvetica, sans-serif", size=8, color="#222222"),
    margin=dict(l=50, r=20, t=30, b=40),
)

_COLORS = {
    "primary": "#11376C",
    "danger":  "#C82823",
    "warning": "#DC7814",
    "success": "#1E964B",
    "gray":    "#8C96A6",
    "accent":  "#0078D4",
    "bar_bg":  "#B8D4E8",
    "bar_fg":  "#6BAED6",
}


def _to_png(fig, width: int = 520, height: int = 280) -> bytes:
    """Plotly figure -> PNG bytes. Falls back to matplotlib if kaleido unavailable."""
    # Attempt kaleido first
    try:
        img = fig.to_image(format="png", width=width, height=height, scale=2)
        if img:
            return img
    except Exception as e:
        logger.info(f"kaleido unavailable, trying matplotlib fallback: {e}")

    # Matplotlib fallback: re-render from Plotly trace data
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib import font_manager as fm

        # Korean font
        _bundled_font = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts" / "NotoSansKR.ttf"
        font_prop = None
        if _bundled_font.exists():
            fm.fontManager.addfont(str(_bundled_font))
            font_prop = fm.FontProperties(fname=str(_bundled_font))
            plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False

        mpl_fig = Figure(figsize=(width / 100, height / 100), dpi=130)
        ax = mpl_fig.add_subplot(111)
        mpl_fig.patch.set_facecolor("white")
        ax.set_facecolor("#FAFBFC")

        # Determine total number of x values for rotation decision
        max_x_count = 0

        for trace in fig.data:
            x = list(trace.x) if hasattr(trace, "x") and trace.x is not None else []
            y = list(trace.y) if hasattr(trace, "y") and trace.y is not None else []
            name = trace.name or ""

            if not x or not y:
                continue

            max_x_count = max(max_x_count, len(x))
            trace_type = trace.type if hasattr(trace, "type") else "scatter"

            # Extract color
            color = None
            if hasattr(trace, "marker") and trace.marker:
                mc = trace.marker.color
                if isinstance(mc, str):
                    color = mc
                elif isinstance(mc, (list, tuple)) and mc:
                    # For bar charts with per-bar colors
                    if all(isinstance(c, str) for c in mc):
                        color = mc  # pass as list

            if trace_type == "bar":
                orientation = getattr(trace, "orientation", None)
                bar_color = color if isinstance(color, (str, type(None))) else (color if isinstance(color, list) else None)

                if orientation == "h":
                    bars = ax.barh(x, y, label=name, color=bar_color, alpha=0.85, edgecolor="white", linewidth=0.5)
                else:
                    bars = ax.bar(x, y, label=name, color=bar_color, alpha=0.85, edgecolor="white", linewidth=0.5)

                # Add value labels on bars if few enough bars
                if len(x) <= 15 and orientation != "h":
                    for bar_obj in bars:
                        h = bar_obj.get_height()
                        if h > 0:
                            ax.text(
                                bar_obj.get_x() + bar_obj.get_width() / 2, h,
                                f"{h:.0f}" if h >= 1 else f"{h:.2f}",
                                ha="center", va="bottom", fontsize=6, color="#555",
                            )
            else:
                line_color = None
                if hasattr(trace, "line") and trace.line and trace.line.color:
                    line_color = trace.line.color
                ax.plot(x, y, label=name, color=line_color, linewidth=2, marker="o", markersize=4)

        # Titles
        layout = fig.layout
        if layout.title and layout.title.text:
            ax.set_title(layout.title.text, fontsize=8, fontweight="bold", pad=6, color="#1A1A2E")
        if layout.xaxis and layout.xaxis.title and layout.xaxis.title.text:
            ax.set_xlabel(layout.xaxis.title.text, fontsize=7, color="#7A8FA6")
        if layout.yaxis and layout.yaxis.title and layout.yaxis.title.text:
            ax.set_ylabel(layout.yaxis.title.text, fontsize=7, color="#7A8FA6")

        # X-axis label rotation for many values
        ax.tick_params(labelsize=6)
        if max_x_count > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6)
        elif max_x_count > 6:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=6)

        if any(t.name for t in fig.data if t.name):
            ax.legend(fontsize=6, loc="upper right", framealpha=0.9)

        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        mpl_fig.tight_layout()

        buf = io.BytesIO()
        mpl_fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=130)
        plt.close(mpl_fig)
        buf.seek(0)
        result = buf.read()
        logger.info(f"matplotlib fallback chart: {len(result):,} bytes")
        return result
    except Exception as e2:
        logger.warning(f"matplotlib fallback failed: {e2}")
        return b""


# ─── 1. Daily Trend (Workers + T-Ward) ──────────────────────────────

def chart_daily_trend(
    metas: list[dict],
    dates: list[str],
    weather_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """Daily worker count bar chart (overlay: total access + T-Ward).
    If weather_df is provided, adds weather icons as annotations on bars.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    date_labels = [f"{d[4:6]}/{d[6:]}" for d in dates]

    workers = [m.get("total_workers_access", 0) for m in metas]
    tward   = [m.get("total_workers_move", 0) for m in metas]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=date_labels, y=workers, name="\ucd9c\uc785\uc778\uc6d0",
            marker_color=_COLORS["bar_bg"], opacity=0.75,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=date_labels, y=tward, name="T-Ward",
            marker_color=_COLORS["primary"], opacity=0.85,
        ),
        secondary_y=False,
    )

    # Weather annotations on top of bars
    if weather_df is not None and not weather_df.empty:
        _WEATHER_ICONS = {"Rain": "\u2602", "Snow": "\u2744", "Sunny": "\u2600", "Unknown": ""}
        weather_map = {}
        for _, row in weather_df.iterrows():
            d = str(row.get("date", "")).replace("-", "")
            weather_map[d] = row.get("weather", "")
        for i, d in enumerate(dates):
            w = weather_map.get(d, "")
            icon = _WEATHER_ICONS.get(w, "")
            if icon:
                fig.add_annotation(
                    x=date_labels[i], y=workers[i],
                    text=icon, showarrow=False,
                    font=dict(size=12), yshift=10,
                )

    fig.update_layout(
        **_PRINT_LAYOUT,
        title=dict(text="\uc77c\ubcc4 \ucd9c\uc785\uc778\uc6d0 \ucd94\uc774", font=dict(size=8)),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="\uc778\uc6d0(\uba85)",
    )

    # Rotate x-axis labels if many dates
    if len(dates) > 10:
        fig.update_xaxes(tickangle=45)

    return _to_png(fig, width=600, height=200)


# ─── 2. EWI/CRE Trend ───────────────────────────────────────────────

def chart_ewi_cre_trend(
    worker_df: pd.DataFrame,
    dates: list[str],
    weather_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """Daily average EWI/CRE line chart with 0.6 threshold line.
    If weather_df provided, overlays average temperature as secondary Y axis.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if "date" not in worker_df.columns:
        return b""

    # 선택한 날짜 범위만 필터링
    if dates:
        worker_df = worker_df[worker_df["date"].isin(dates)]

    daily = (
        worker_df.groupby("date")
        .agg(avg_ewi=("ewi", "mean"), avg_cre=("cre", "mean"),
             high_cre=("cre", lambda x: (x.fillna(0) >= CRE_HIGH).sum()))
        .reset_index()
        .sort_values("date")
    )

    date_labels = [f"{d[4:6]}/{d[6:]}" for d in daily["date"]]

    has_weather = (
        weather_df is not None
        and not weather_df.empty
        and "temp_max" in weather_df.columns
    )

    if has_weather:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=date_labels, y=daily["avg_ewi"], name="\ud3c9\uade0 EWI",
        mode="lines+markers",
        line=dict(color=_COLORS["primary"], width=2.5),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=date_labels, y=daily["avg_cre"], name="\ud3c9\uade0 CRE",
        mode="lines+markers",
        line=dict(color=_COLORS["danger"], width=2.5),
        marker=dict(size=7),
    ))
    fig.add_hline(
        y=0.6, line_dash="dash", line_color="#E74C3C",
        annotation_text="\uace0\uc704\ud5d8 \uc784\uacc4(0.6)",
        annotation_position="top left",
        annotation_font_size=9,
    )

    # Temperature overlay on secondary Y axis
    if has_weather:
        weather_map = {}
        for _, row in weather_df.iterrows():
            d = str(row.get("date", "")).replace("-", "")
            t_max = row.get("temp_max")
            t_min = row.get("temp_min")
            if t_max is not None and t_min is not None:
                try:
                    weather_map[d] = (float(t_max) + float(t_min)) / 2
                except (ValueError, TypeError):
                    pass
        temp_vals = [weather_map.get(d) for d in daily["date"]]
        if any(v is not None for v in temp_vals):
            fig.add_trace(
                go.Scatter(
                    x=date_labels,
                    y=temp_vals,
                    name="평균기온(°C)",
                    mode="lines+markers",
                    line=dict(color=_COLORS["warning"], width=1.5, dash="dot"),
                    marker=dict(size=5, symbol="diamond"),
                    opacity=0.7,
                ),
                secondary_y=True,
            )
            fig.update_yaxes(
                title_text="기온(°C)", secondary_y=True,
                showgrid=False,
            )

    fig.update_layout(
        **_PRINT_LAYOUT,
        title=dict(text="\uc77c\ubcc4 EWI / CRE \ucd94\uc774", font=dict(size=8)),
        yaxis_title="\uc9c0\ud45c\uac12",
        yaxis_range=[0, max(1.0, daily["avg_ewi"].max() * 1.2)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if len(date_labels) > 10:
        fig.update_xaxes(tickangle=45)

    return _to_png(fig, width=600, height=200)


# ─── 3. CRE Distribution Histogram ──────────────────────────────────

def chart_cre_distribution(worker_df: pd.DataFrame) -> bytes:
    """CRE histogram with >= CRE_HIGH region highlighted in red (constants.py 기준)."""
    import plotly.graph_objects as go

    if "cre" not in worker_df.columns:
        return b""

    cre = worker_df["cre"].dropna()
    bins = np.arange(0, 1.05, 0.05)

    hist_vals, hist_edges = np.histogram(cre, bins=bins)
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    colors = [_COLORS["danger"] if c >= CRE_HIGH else _COLORS["primary"] for c in bin_centers]

    high_count = int((cre >= CRE_HIGH).sum())
    total = len(cre)

    fig = go.Figure(go.Bar(
        x=[f"{c:.2f}" for c in bin_centers],
        y=hist_vals,
        marker_color=colors,
    ))

    fig.update_layout(
        **_PRINT_LAYOUT,
        title=dict(
            text=f"CRE \ubd84\ud3ec (\uace0\uc704\ud5d8 \u22650.6: {high_count}\uba85 / {total}\uba85, {high_count/total*100:.1f}%)",
            font=dict(size=8),
        ),
        xaxis_title="CRE \uac12",
        yaxis_title="\uc791\uc5c5\uc790 \uc218",
    )

    if len(bin_centers) > 10:
        fig.update_xaxes(tickangle=45)

    return _to_png(fig, width=600, height=200)


# ─── 4. Company Risk Comparison ──────────────────────────────────────

def chart_company_risk(worker_df: pd.DataFrame, top_n: int = 10) -> bytes:
    """Horizontal bar chart of average CRE by company (top N, >= 5 workers)."""
    import plotly.graph_objects as go

    if "company_name" not in worker_df.columns or "cre" not in worker_df.columns:
        return b""

    comp = (
        worker_df.groupby("company_name")
        .agg(avg_cre=("cre", "mean"), count=("cre", "count"))
        .reset_index()
    )
    # Filter: 5+ workers, exclude 미확인
    comp = comp[(comp["count"] >= 5) & (comp["company_name"] != "\ubbf8\ud655\uc778")]
    comp = comp.nlargest(top_n, "avg_cre")

    if comp.empty:
        return b""

    comp["short_name"] = comp["company_name"].str[:15]

    colors = [
        _COLORS["danger"] if v >= 0.5
        else _COLORS["warning"] if v >= 0.35
        else _COLORS["success"]
        for v in comp["avg_cre"]
    ]

    fig = go.Figure(go.Bar(
        y=comp["short_name"],
        x=comp["avg_cre"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f} ({c}\uba85)" for v, c in zip(comp["avg_cre"], comp["count"])],
        textposition="auto",
        textfont=dict(size=8),
    ))

    fig.update_layout(
        **_PRINT_LAYOUT,
        title=dict(
            text=f"\uc5c5\uccb4\ubcc4 \ud3c9\uade0 CRE (\uc0c1\uc704 {len(comp)}\uac1c, 5\uba85 \uc774\uc0c1)",
            font=dict(size=8),
        ),
        xaxis_title="\ud3c9\uade0 CRE",
        yaxis=dict(autorange="reversed"),
        height=160,
    )

    return _to_png(fig, width=600, height=200)


# ─── 5. Fatigue vs CRE Scatter ──────────────────────────────────────

def chart_fatigue_vs_cre(worker_df: pd.DataFrame) -> bytes:
    """Fatigue vs CRE scatter — compound risk visualization."""
    import plotly.graph_objects as go

    if "fatigue_score" not in worker_df.columns or "cre" not in worker_df.columns:
        return b""

    df = worker_df[["fatigue_score", "cre"]].dropna()
    if df.empty:
        return b""

    both_high = ((df["fatigue_score"] >= 0.6) & (df["cre"] >= CRE_HIGH)).sum()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["fatigue_score"], y=df["cre"],
        mode="markers",
        marker=dict(
            size=5, opacity=0.5,
            color=df["cre"],
            colorscale=[[0, _COLORS["success"]], [0.5, _COLORS["warning"]], [1.0, _COLORS["danger"]]],
        ),
        hoverinfo="skip",
    ))

    # Danger zone
    fig.add_shape(type="rect", x0=0.6, y0=0.6, x1=1.0, y1=1.0,
                  fillcolor="rgba(231,76,60,0.08)", line=dict(width=0))
    fig.add_hline(y=0.6, line_dash="dot", line_color="#ccc")
    fig.add_vline(x=0.6, line_dash="dot", line_color="#ccc")

    fig.update_layout(
        **_PRINT_LAYOUT,
        title=dict(
            text=f"\ud53c\ub85c\ub3c4 vs CRE (\ubcf5\ud569 \uace0\uc704\ud5d8: {both_high}\uba85)",
            font=dict(size=8),
        ),
        xaxis_title="\ud53c\ub85c\ub3c4",
        yaxis_title="CRE (\uc704\ud5d8\ub178\ucd9c)",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )

    return _to_png(fig, width=520, height=200)
