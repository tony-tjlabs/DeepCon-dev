"""
DeepCon 메트릭 엔진 — 공개 API.

사용:
    from src.metrics import (
        get_time_breakdown, get_risk_summary, get_daily_kpi, get_company_ranking,
        RISK_THRESHOLDS, TIME_TARGETS,
        apply_default, apply_tward_reliable,
    )
"""
from src.metrics.constants import (
    TIME_TARGETS,
    RISK_THRESHOLDS,
    CONGESTION_GRADES,
    TIME_BREAKDOWN_CATEGORIES,
    TIME_BREAKDOWN_LABELS,
    CONSISTENCY_TOLERANCE,
)
from src.metrics.filters import (
    keep_has_tward, keep_ewi_reliable, keep_positive_work_minutes,
    keep_not_missing_exit, exclude_unidentified,
    apply_default, apply_tward_reliable,
    safe_sum, safe_mean,
)
from src.metrics.daily import (
    TimeBreakdown, RiskSummary, DailyKPI,
    get_time_breakdown, get_risk_summary,
    get_company_ranking, get_daily_kpi,
)

__all__ = [
    # constants
    "TIME_TARGETS", "RISK_THRESHOLDS", "CONGESTION_GRADES",
    "TIME_BREAKDOWN_CATEGORIES", "TIME_BREAKDOWN_LABELS",
    "CONSISTENCY_TOLERANCE",
    # filters
    "keep_has_tward", "keep_ewi_reliable", "keep_positive_work_minutes",
    "keep_not_missing_exit", "exclude_unidentified",
    "apply_default", "apply_tward_reliable",
    "safe_sum", "safe_mean",
    # daily aggregates
    "TimeBreakdown", "RiskSummary", "DailyKPI",
    "get_time_breakdown", "get_risk_summary",
    "get_company_ranking", "get_daily_kpi",
]
