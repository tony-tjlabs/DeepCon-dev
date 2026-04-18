"""
Weekly 패키지
=============
주간 리포트 탭 모듈 분할.
"""
from src.dashboard._legacy.weekly.summary import (
    render_weekly_kpi,
    render_weekly_site_status,
    render_daily_trend,
    render_ewi_cre_trend,
)
from src.dashboard._legacy.weekly.space_analysis import (
    render_weekly_space,
    render_weekly_company,
    render_weekly_safety,
)
from src.dashboard._legacy.weekly.patterns import (
    render_day_of_week_analysis,
    render_shift_trend,
)

__all__ = [
    # summary
    "render_weekly_kpi",
    "render_weekly_site_status",
    "render_daily_trend",
    "render_ewi_cre_trend",
    # space_analysis
    "render_weekly_space",
    "render_weekly_company",
    "render_weekly_safety",
    # patterns
    "render_day_of_week_analysis",
    "render_shift_trend",
]
