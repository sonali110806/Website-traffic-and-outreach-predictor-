# Website Traffic Forecasting — Results

## 1. Objective
Develop a machine learning model to predict website traffic 30 minutes into the future using historical traffic patterns and related features.

## 2. Dataset
- Total original records: 43,200
- Input features: 13
- Forecast horizon: 30 minutes
- Train-validation-test split: 60%-20%-20%, in chronological order

## 3. Model
- Algorithm: XGBoost Regressor
- Baseline: Current traffic used as the prediction for traffic 30 minutes later

## 4. Test Results

| Metric | XGBoost | Baseline |
|---|---:|---:|
| MAE | 150.67 | 180.52 |
| RMSE | 203.95 | 239.70 |

## 5. Conclusion
XGBoost achieved lower MAE and RMSE than the baseline on the test dataset. Test MAE improved by approximately 16.5%, and test RMSE improved by approximately 14.9%.

The results are promising for forecasting traffic on this synthetic dataset. Performance on real-world website traffic requires further validation.
