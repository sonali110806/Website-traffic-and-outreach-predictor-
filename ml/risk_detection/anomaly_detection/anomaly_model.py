
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = [
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
]


def build_anomaly_model(contamination=0.01, random_state=42):
    """Create an Isolation Forest model for anomaly detection."""
    return IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
 )
