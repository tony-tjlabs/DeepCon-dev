"""
Daily Tab Stub — Backward Compatibility
=======================================
기존 import 경로 유지를 위한 스텁 파일.

사용:
    from src.dashboard.daily_tab import render_daily_tab

실제 구현은 src/dashboard/daily/ 패키지에 있음.
"""
from __future__ import annotations

# 패키지에서 re-export
from src.dashboard.daily import render_daily_tab

__all__ = ["render_daily_tab"]
