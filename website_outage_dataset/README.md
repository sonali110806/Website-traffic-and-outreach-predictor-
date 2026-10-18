# Website Traffic / Outage Risk Predictor — Synthetic Dataset

## Scope
- Synthetic minute-level telemetry covering 30 days (43,200 rows).
- Designed for traffic forecasting, anomaly detection, outage/degradation risk classification, and dashboard demonstrations.
- All values are simulated, not from a real production website.

## Files
- `website_traffic_outage_synthetic_full.csv`: complete dataset with raw-like metrics, labels, time features, and causal lag/rolling features.
- `traffic_forecasting_dataset.csv`: compact dataset for Member 2's forecasting module.
- `outage_risk_dataset.csv`: compact dataset for Member 3's risk/anomaly module.
- `data_dictionary.csv`: field descriptions and usage notes.
- `generate_dataset.py`: reproducible generator (seed 42).

## Suggested usage
- Forecasting: predict `requests_per_minute` at future horizons. Use chronological train/test splits. For a true 30-minute-ahead target, create target with `shift(-30)` and drop rows where target is missing.
- Anomaly detection: use metrics such as traffic, CPU, memory, latency, error rate, network usage, and database connections. `anomaly_label` is a synthetic evaluation label; do not include it as a model input.
- Outage risk: target `outage_within_30m` (binary) and exclude rows where `risk_label_valid == 0`. Do not use `website_status`, `event_type`, `anomaly_label`, or `outage_within_30m` as input features. This label is defined by whether a degraded/outage state begins within the next 30 minutes.
- The data includes a small number of explicit synthetic incidents. Model results on synthetic data demonstrate pipeline behavior, not real-world performance.

## Important caveat
Because labels are simulated and derived from programmed scenarios, high test metrics do not establish that a model can predict real production outages. Validate on real telemetry before operational use.
