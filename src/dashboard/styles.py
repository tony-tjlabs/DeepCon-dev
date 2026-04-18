"""
DeepCon 대시보드 CSS 스타일 — 엔터프라이즈 다크 테마
=====================================================
★ v2 (Session 4): SK하이닉스 임원 프레젠테이션 수준 품질
  - 대비 강화 (WCAG AA+ 준수)
  - 3단계 폰트 계층 (Primary/Secondary/Detail)
  - 통일된 spacing/radius 시스템
  - hover 효과 강화 (box-shadow glow)
  - Plotly 차트 공유 설정 상수
"""

# ─── 디자인 토큰 (파이썬 코드에서 참조용) ─────────────────────────────

COLORS = {
    "bg":        "#0D1B2A",
    "card_bg":   "#1A2A3A",
    "border":    "#2A3A4A",
    "primary":   "#1A3A5C",
    "accent":    "#00AEEF",
    "text":      "#D5E5FF",
    "text_muted": "#9AB5D4",
    "text_dim":  "#6A7A95",
    "success":   "#00C897",
    "warning":   "#FFB300",
    "danger":    "#FF4C4C",
    "confined":  "#FF6B35",
    # ── 확장 semantic 토큰 (M3-B, 하드코딩 정리) ───
    "bg_deep":      "#0A1520",   # 사이드바, 최하단 배경
    "bg_card_sub":  "#111820",   # 카드 내부/insight 배경
    "bg_chart":     "#111820",   # Plotly plot_bgcolor
    "text_bright":  "#C8D6E8",   # KPI 숫자 등 highlight
    "text_dimmer":  "#7A8FA6",   # 캡션/보조 텍스트
    "text_muted2":  "#4A6A8A",   # 캡션 계열 진한 버전
    "text_ghost":   "#3A4A5A",   # 매우 옅은 텍스트
    "text_deepest": "#2A3A4A",   # 최하위 계층
    "grid":         "#2A3A4A",   # 차트 그리드 선
    # ── 상태/레벨 뱃지 (integrity 등) ───
    "danger_soft":  "#FF8C00",   # high severity (critical 과 구분)
    "low_active":   "#0078AA",
    "gap_dark":     "#4A5A6A",
    "access":       "#4A6A8A",
    "pass":         "#00C897",   # success alias
    "conditional":  "#FFB300",   # warning alias
    "needs_fix":    "#FF8C00",   # danger_soft alias
    "block":        "#FF4C4C",   # danger alias
    # ── 활동/Journey block_type 매핑 ───
    "block_gate":   "#6B7280",
    "block_work":   "#3B82F6",
    "block_rest":   "#10B981",
    "block_transit":"#9CA3AF",
    "block_admin":  "#8B5CF6",
    "block_unknown":"#4B5563",
}


def alpha_hex(color: str, alpha_pct: int) -> str:
    """hex 색상에 알파 채널 추가 (e.g. '#FF4C4C' + 18 → '#FF4C4C2E').

    Streamlit은 알파를 포함한 8자리 hex를 인라인 CSS에서 허용한다.
    """
    if not color or not color.startswith("#") or len(color) != 7:
        return color
    a = max(0, min(255, round(alpha_pct * 255 / 100)))
    return f"{color}{a:02X}"

# ═══════════════════════════════════════════════════════════════════════
# CHART_COLORS — 차트(도넛·바·라인·히트맵 등)용 개념 단위 색상 팔레트
# ───────────────────────────────────────────────────────────────────────
# 모든 탭·차트가 이 딕셔너리만 참조하여 **전사 색상 일관성**을 확보한다.
# 직접 hex 하드코딩 금지. 새 개념이 필요하면 여기에 추가.
# ═══════════════════════════════════════════════════════════════════════
CHART_COLORS = {
    # ── 작업시간 카테고리 ─────────────────────────
    "work_zone":   "#00AEEF",  # accent: 작업공간 (생산 활동)
    "transit":     "#9AB5D4",  # text_muted: 이동
    "rest":        "#00C897",  # success: 휴게
    "gap":         "#4A5A6A",  # gap_dark: BLE 음영 (회색톤)
    "other":       "#6A7A95",  # text_dim: 기타/미분류

    # ── 3대 지표 ────────────────────────────────
    "ewi":         "#00AEEF",  # accent: EWI (생산성)
    "cre":         "#FFB300",  # warning: CRE (위험노출)
    "sii":         "#FF8C42",  # SII 전용 (CRE와 구분)
    "fatigue":     "#FF6B35",  # confined 계열: 피로도

    # ── 위험 수준 (Alert / Insight 공통) ────────
    "critical":    "#FF4C4C",  # danger
    "high":        "#FF8C42",  # SII 계열 (high risk)
    "medium":      "#FFB300",  # warning
    "low":         "#6A7A95",  # text_dim
    "info":        "#00AEEF",  # accent (정보성)

    # ── 공간 유형 ──────────────────────────────
    "confined_space": "#FF6B35",  # confined
    "high_voltage":   "#FF4C4C",  # danger
    "gate":           "#A78BFA",  # purple (출입공간)
    "rest_area":      "#00C897",  # success
    "work_area":      "#00AEEF",  # accent

    # ── Raw vs 보정 ─────────────────────────────
    "raw":         "#9AB5D4",  # 원본 (흐린 파랑)
    "corrected":   "#00AEEF",  # 보정 (강조 파랑)

    # ── T-Ward / 타각기 / 출입 ──────────────────
    "tward":       "#00C897",  # T-Ward 착용 (success)
    "access":      "#4A6A8A",  # 출입 전체 (회청)

    # ── 활동 수준 ───────────────────────────────
    "high_active": "#FF4C4C",  # 고활성 (danger)
    "low_active":  "#0078AA",  # 저활성 (dark blue)
    "standby":     "#9AB5D4",  # 대기 (text_muted)
    "deep_inact":  "#4A5A6A",  # DEEP_INACTIVE

    # ── gap-fill 신뢰도 ─────────────────────────
    "gap_high":    "#00AEEF",  # 고신뢰 보정
    "gap_medium":  "#FFB300",
    "gap_low":     "#FF4C4C",
    "gap_none":    "#6A7A95",

    # ── 업체/작업자 랭킹용 Qualitative Palette ──
    # 여러 시리즈가 필요할 때 순환 사용
    "palette": [
        "#00AEEF",  # accent
        "#00C897",  # success
        "#FFB300",  # warning
        "#FF8C42",  # sii
        "#A78BFA",  # purple
        "#FF6B35",  # confined
        "#4A6A8A",  # access
        "#0078AA",  # low_active
        "#9AB5D4",  # text_muted
        "#FF4C4C",  # danger (마지막 - 주의 환기용)
    ],

    # ── 연속 스케일 (히트맵) ─────────────────────
    # 낮음(정상/좋음) → 중간(주의) → 높음(위험)
    "scale_risk":        ["#00C897", "#FFB300", "#FF4C4C"],
    # 단색 그라데이션 (활동량 등)
    "scale_activity":    ["#1A3A5C", "#00AEEF", "#D5E5FF"],
    # 커버리지 (음영 → 충분)
    "scale_coverage":    ["#4A5A6A", "#FFB300", "#00C897"],
}

SPACING = {"xs": "4px", "sm": "8px", "md": "12px", "lg": "16px", "xl": "20px", "xxl": "24px"}
RADIUS  = {"sm": "6px", "md": "10px", "lg": "12px"}

# ─── Plotly 공유 설정 (모든 차트에서 import) ──────────────────────────

PLOTLY_DARK = dict(
    paper_bgcolor="#1A2A3A",
    plot_bgcolor="#111820",
    font_color="#D5E5FF",
    margin=dict(l=15, r=15, t=50, b=15),
)
PLOTLY_LEGEND = dict(font=dict(color="#D5E5FF", size=11))


# ─── CSS ──────────────────────────────────────────────────────────

DARK_CSS = """
<style>
/* ── 전역 ── */
:root {
    --bg:          #0D1B2A;
    --card-bg:     #1A2A3A;
    --border:      #2A3A4A;
    --primary:     #1A3A5C;
    --accent:      #00AEEF;
    --text:        #D5E5FF;
    --text-muted:  #9AB5D4;
    --text-dim:    #6A7A95;
    --success:     #00C897;
    --warning:     #FFB300;
    --danger:      #FF4C4C;
    --confined:    #FF6B35;
}

[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"]          { background: #0A1520; border-right: 1px solid var(--border); }
[data-testid="stHeader"]           { background: transparent; }

/* ── 타이포그래피 — 3단계 계층 ── */
h1, h2, h3, h4 { color: var(--text) !important; }
p, li, span, label, div { color: var(--text); }
.st-emotion-cache-16idsys p { font-size: 14px; }

/* ── 메트릭 카드 ── */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--accent);
    box-shadow: 0 6px 20px rgba(0, 174, 239, 0.12);
}
/* Tier 1: 핵심 KPI 숫자 */
.metric-value { font-size: 2.0rem; font-weight: 700; color: var(--accent); }
/* Tier 2: 보조 KPI */
.metric-value-secondary { font-size: 1.5rem; font-weight: 700; }
/* Tier 3: 상세 수치 */
.metric-value-detail { font-size: 1.0rem; font-weight: 600; }

.metric-label { font-size: 0.82rem; color: var(--text-muted); margin-top: 6px; }
.metric-delta { font-size: 0.85rem; margin-top: 4px; }
.metric-delta.up   { color: var(--success); }
.metric-delta.down { color: var(--danger); }

/* ── 상태 뱃지 ── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 14px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid;
}
.badge-success  { background: rgba(0,200,151,0.20); color: var(--success); border-color: var(--success); }
.badge-warning  { background: rgba(255,179,0,0.20);  color: var(--warning); border-color: var(--warning); }
.badge-danger   { background: rgba(255,76,76,0.20);  color: var(--danger);  border-color: var(--danger); }
.badge-info     { background: rgba(0,174,239,0.20);  color: var(--accent);  border-color: var(--accent); }
.badge-confined { background: rgba(255,107,53,0.20); color: var(--confined);border-color: var(--confined); }

/* ── 섹션 헤더 ── */
.section-header {
    border-left: 4px solid var(--accent);
    padding-left: 14px;
    margin: 28px 0 16px 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.3px;
}

/* ── 인사이트/알림 카드 ── */
.insight-card {
    background: #111820;
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 0;
}
.insight-title {
    font-size: 0.92rem;
    color: var(--text);
    font-weight: 600;
    margin-bottom: 6px;
}
.insight-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.45;
}
.insight-detail {
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 6px;
}

/* ── 파이프라인 상태 카드 ── */
.pipeline-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.pipeline-card.unprocessed { border-left: 4px solid var(--warning); }
.pipeline-card.processed   { border-left: 4px solid var(--success); }

/* ── 테이블 ── */
[data-testid="stDataFrame"] {
    background: var(--card-bg) !important;
    border-radius: 8px;
}

/* ── 버튼 ── */
.stButton > button {
    background: linear-gradient(135deg, #1A3A5C, #0088CC);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    opacity: 0.9;
    box-shadow: 0 4px 12px rgba(0, 174, 239, 0.2);
}

/* ── 선택박스 ── */
.stSelectbox > div, .stMultiSelect > div {
    background: var(--card-bg);
    border-color: var(--border);
}

/* ── Expander (데이터 해석) ── */
.streamlit-expanderHeader {
    background: #0D1520 !important;
    border-radius: 8px;
    font-size: 0.88rem;
}

/* ── 탭 ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 600;
}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(DARK_CSS, unsafe_allow_html=True)


# ─── 헬퍼 함수 ──────────────────────────────────────────────────────

def metric_card(label: str, value: str, delta: str = "", delta_up: bool = True, color: str = "") -> str:
    """Tier 1 메트릭 카드."""
    delta_class = "up" if delta_up else "down"
    delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    color_style = f"color:{color};" if color else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="{color_style}">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>"""


def metric_card_sm(label: str, value: str, color: str = "") -> str:
    """Tier 2 보조 메트릭 카드 (공간/업체 카드 등)."""
    color_style = f"color:{color};" if color else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value-secondary" style="{color_style}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def insight_card(title: str, description: str, detail: str = "",
                 severity: str = "info") -> str:
    """인사이트/알림 카드."""
    severity_colors = {
        "critical": "#FF4C4C", "high": "#FF8C00",
        "medium": "#FFB300", "low": "#00AEEF",
        "info": "#00AEEF", "success": "#00C897",
    }
    border_color = severity_colors.get(severity, "#00AEEF")
    detail_html = f'<div class="insight-detail">{detail}</div>' if detail else ""
    return f"""
    <div class="insight-card" style="border-left-color:{border_color}">
        <div class="insight-title">{title}</div>
        <div class="insight-desc">{description}</div>
        {detail_html}
    </div>"""


def section_header(title: str) -> str:
    return f'<div class="section-header">{title}</div>'


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def sub_header(title: str) -> str:
    """서브 섹션 헤더 (차트 위 등)."""
    return (
        f'<div style="font-size:0.95rem; font-weight:600; color:#D5E5FF; '
        f'margin:16px 0 8px 0;">{title}</div>'
    )
