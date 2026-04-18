"""
DeepCon Domain Packs
====================
도메인별 설정, 지식, 프롬프트, 규칙을 분리하여 관리하는 패키지.

각 도메인 팩은 다음을 포함:
  - config.yaml: 도메인 설정 (임계값, 가중치, 상수)
  - locus_attrs.csv: Locus 확장 속성 (hazard_level, confined_space 등)
  - prompts.py: LLM 프롬프트 템플릿
  - rules.py: 알림/판정/분류 규칙

도메인:
  - construction: 건설현장 (EWI/CRE, 단독작업, 밀폐공간)
  - retail: 리테일 (DC/AST, 전환율)
  - airport: 공항 (Zone 메트릭, 혼잡도)
"""
