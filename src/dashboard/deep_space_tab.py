"""
Deep Space 분석 탭 (라우터)
==========================
학습된 Deep Space Transformer 모델을 활용한 Spatial AI 분석.

8가지 분석 기능:
  1. Agentic AI -- Intelligence + Deep Space 통합 예측 (혼잡/병목/안전/생산성)
  2. 현장 시뮬레이션 -- Entity/Locus 통합 뷰
  3. 개요 -- 모델 KPI + 전이 히트맵 + 인사이트
  4. 이동 예측 -- 작업자의 다음 이동 장소 Top-K 예측
  5. 공간 예측 -- Locus별 미래 상태 예측
  6. 이상 이동 탐지 -- 비정상 이동 패턴 감지
  7. 공간 관계 -- Locus 간 관계 시각화
  8. 학습 현황 -- 모델 성능 및 학습 이력

실제 구현은 src/dashboard/deep_space/ 패키지에 분산.
"""
from __future__ import annotations

import streamlit as st

from src.dashboard.styles import COLORS
from src.dashboard.deep_space import (
    load_model,
    get_available_dates,
    render_overview,
    render_training,
    render_prediction,
    render_simulation,
    render_locus_prediction,
    render_anomaly,
    render_spatial_relations,
    render_agentic,
)


def render_deep_space_tab(sector_id: str):
    """Deep Space 분석 탭 메인 (라우터)."""
    st.markdown(
        f"<div style='display:flex; align-items:baseline; gap:10px; margin-bottom:2px;'>"
        f"<h2 style='color:{COLORS['text']}; margin:0;'>Deep Space</h2>"
        f"<span style='background:#1a7a4a; color:#4ade80; font-size:0.72rem; font-weight:700; "
        f"padding:2px 8px; border-radius:12px; letter-spacing:0.5px;'>v2</span>"
        f"<span style='color:{COLORS['text_muted']}; font-size:0.78rem;'>"
        f"Top-1 60.1% &middot; Top-3 82.5% &middot; 3.4M params &middot; 40일 학습</span>"
        f"</div>"
        f"<p style='color:{COLORS['text_muted']}; font-size:0.88rem; margin-bottom:12px;'>"
        f"Spatial AI Foundation &mdash; "
        f"Agentic AI / 시뮬레이션 / 개요 / 예측 / 공간예측 / 이상탐지 / 공간관계 / 학습</p>",
        unsafe_allow_html=True,
    )

    model, tokenizer = load_model(sector_id)
    if model is None:
        st.warning("Deep Space 모델이 없습니다. 파이프라인 탭에서 학습을 먼저 진행하세요.")
        return

    dates = get_available_dates(sector_id)

    tabs = st.tabs([
        "Agentic AI",
        "현장 시뮬레이션",
        "개요",
        "이동 예측",
        "공간 예측",
        "이상 이동 탐지",
        "공간 관계",
        "학습 현황",
    ])

    with tabs[0]:
        with st.expander("Agentic AI란?", expanded=False):
            st.markdown(
                "**Intelligence Engine (EWI/CRE/SII)** + **Deep Space (Transformer)** 을 통합한 예측 시스템.\n\n"
                "- **혼잡도 예측**: 현재 인원 + 이동 예측 → 공간별 미래 혼잡도\n"
                "- **병목 예측**: 유입 > 유출 불균형 → 사람이 쌓이는 공간 감지\n"
                "- **안전 위험 예측**: 위험 구역 진입 예측 + CRE/피로도 복합 분석\n"
                "- **생산성 예측**: 피로 누적, 과다 이동, 작업공간 저활용 → EWI 하락 예측\n\n"
                "Deep Space가 **이동 패턴**을 예측하고, Intelligence Engine이 **작업자 상태**를 분석하여,\n"
                "실시간으로 **공간을 더 잘 관리**하기 위한 액션을 제안합니다."
            )
        render_agentic(model, tokenizer, sector_id, dates)

    with tabs[1]:
        with st.expander("현장 시뮬레이션이란?", expanded=False):
            st.markdown(
                "**Entity(작업자)와 Locus(공간)를 통합**하여 현장 전체를 한 눈에 보여줍니다.\n\n"
                "- **KPI 카드**: 총 인원, 작업/이동/휴식 중, 혼잡 공간, 안전 경고\n"
                "- **Locus 상태 맵 (Treemap)**: 공간별 인원 분포 + 혼잡도 색상\n"
                "- **이동 흐름 (Sankey)**: 현재 위치 -> 예측 다음 위치\n"
                "- **예측 요약**: 혼잡/병목/안전 위험 예상 공간\n"
                "- **시간대별 추이**: 주요 공간의 시간대별 인원 변화"
            )
        render_simulation(model, tokenizer, sector_id, dates)

    with tabs[2]:
        with st.expander("개요란?", expanded=False):
            st.markdown(
                "Deep Space 모델의 **전체 상태**를 한눈에 보여줍니다.\n\n"
                "- **모델 KPI**: 정확도, Vocab 크기, 학습 데이터, 모델 상태\n"
                "- **전이 확률 히트맵**: Locus 간 이동 확률을 시각화\n"
                "- **유입/유출 분석**: 특정 공간의 유입/유출 Top-5\n"
                "- **인사이트**: 데이터에서 자동 추출 (LLM 사용 안 함)"
            )
        render_overview(model, tokenizer, sector_id, dates)

    with tabs[3]:
        with st.expander("이동 예측이란?", expanded=False):
            st.markdown(
                "작업자의 **과거 이동 경로**를 분석하여 **다음에 어디로 이동할지** Top-5로 예측합니다.\n\n"
                "- Deep Space 모델은 수만 명의 이동 패턴을 학습하여 공간 간 전환 확률을 파악합니다\n"
                "- **활용**: 혼잡 예측, 호이스트 배차 최적화, 이동 동선 사전 안내\n"
                "- **예측 vs 실제 비교**를 통해 모델 성능을 실시간 검증할 수 있습니다\n"
                "- **신뢰도 라벨**: High/Medium/Low로 예측 품질을 즉시 파악"
            )
        render_prediction(model, tokenizer, sector_id, dates)

    with tabs[4]:
        with st.expander("공간 예측이란?", expanded=False):
            st.markdown(
                "전체 작업자의 다음 이동을 예측하여 **각 공간(Locus)별 미래 상태**를 예측합니다.\n\n"
                "- **혼잡도**: 예측 인원 / 공간 수용력 -- 70% 이상 시 혼잡 경고\n"
                "- **병목 위험**: (유입 - 유출) / 현재 인원 -- 양수면 사람이 몰리는 곳\n"
                "- **위험도 지수**: 혼잡도 x 공간 위험등급 -- 위험 구역에 사람이 많을수록 위험\n"
                "- **작업 공간 가동률**: 작업 구역에 배치된 인원 비율\n"
                "- **활용**: 혼잡 사전 대응, 호이스트/클라이머 배차 최적화, 작업 재배치 계획"
            )
        render_locus_prediction(model, tokenizer, sector_id, dates)

    with tabs[5]:
        with st.expander("이상 이동 탐지란?", expanded=False):
            st.markdown(
                "각 작업자의 이동 경로에서 **모델이 예측하지 못한 비정상 이동**을 감지합니다.\n\n"
                "- **Surprisal 점수**: 모델이 예상하지 못할수록 높은 점수 (= 비정상)\n"
                "- **위험도 4단계**: Critical / High / Medium / Low로 분류\n"
                "- **임계값 슬라이더**: sigma 배수를 조정하여 민감도를 제어합니다\n"
                "- **활용**: 안전 모니터링, 비인가 구역 진입 감지, 피로 누적 후 비정상 행동 탐지"
            )
        render_anomaly(model, tokenizer, sector_id, dates)

    with tabs[6]:
        with st.expander("공간 관계란?", expanded=False):
            st.markdown(
                "Deep Space 모델이 학습한 **공간의 벡터 표현**을 2D로 시각화합니다.\n\n"
                "- 가까운 공간 = **이동 패턴이 유사한 공간** (물리적 거리와 다를 수 있음)\n"
                "- 같은 클러스터 = **기능적으로 연결된 공간** (예: FAB 3F<->4F<->5F)\n"
                "- **유사도 검색**: 특정 공간과 이동 패턴이 비슷한 공간을 찾아줍니다\n"
                "- **활용**: 공간 재배치 계획, 동선 최적화, 기능적 구역 재분류"
            )
        render_spatial_relations(model, tokenizer, sector_id)

    with tabs[7]:
        with st.expander("학습 현황이란?", expanded=False):
            st.markdown(
                "Deep Space Transformer 모델의 **학습 과정과 성능 지표**를 보여줍니다.\n\n"
                "- **Loss**: 낮을수록 모델이 이동 패턴을 잘 학습한 것\n"
                "- **Top-K 정확도**: 다음 이동 장소를 K개 후보 중 맞출 확률\n"
                "- 모델은 BERT 스타일 MLM(Masked Language Model)으로 학습됩니다\n"
                "- Journey(이동 경로) = 문장, Locus(장소) = 단어로 취급"
            )
        render_training(sector_id)
