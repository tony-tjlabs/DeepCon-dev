"""
건설현장 도메인 팩 — SK하이닉스 반도체 클러스터
==============================================
건설현장의 안전/생산성 분석을 위한 도메인 지식, 설정, 규칙 모음.

핵심 지표:
  - EWI (Effective Work Intensity): 유효작업집중도
  - CRE (Construction Risk Exposure): 건설위험노출도
  - SII (Spatial Intensity Index): 공간강도지수

핵심 기능:
  - 단독작업 감지 (밀폐공간, 고압전 구역)
  - Journey 보정 (BLE 노이즈 플리커 제거)
  - 공간 위험도 분류 (hazard_level, confined_space)

파일:
  - config.yaml: EWI/CRE 임계값, 단독작업 기준, 근무시간 설정
  - locus_attrs.csv: 58개 Locus의 건설 도메인 속성
  - prompts.py: LLM 프롬프트 템플릿 (현장 맥락 + 분석 요청)
  - rules.py: 알림 조건, 등급 판정 규칙
"""
from pathlib import Path

DOMAIN_PACK_DIR = Path(__file__).parent
CONFIG_FILE = DOMAIN_PACK_DIR / "config.yaml"
LOCUS_ATTRS_FILE = DOMAIN_PACK_DIR / "locus_attrs.csv"
