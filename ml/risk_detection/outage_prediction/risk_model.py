
from xgboost import XGBClassifier

RISK_FEATURES = [
    "requests_per_minute",
    "active_users",
    "cpu_usage_pct",
    "memory_usage_pct",
    "latency_ms",
    "error_rate_pct",
    "http_5xx_count",
    "network_usage_mbps",
    "database_connections",
    "traffic_growth_pct_5m",
    "cpu_change_5m",
    "latency_change_5m",
    "error_rate_change_5m",
    "traffic_rolling_mean_5m",
    "traffic_rolling_mean_15m",
]


def build_risk_model(scale_pos_weight, random_state=42):
    """Create an XGBoost classifier for 30-minute outage risk."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
