# Model Governance

## Promotion Criteria
A model must achieve:
- F1 score >= 0.88
- AUC-ROC >= 0.92
- Latency p99 < 200ms
- Drift score < 0.15

## Staging -> Production
1. Automated validation in staging
2. 10% traffic A/B test (challenger)
3. Statistical significance test
4. Full promotion if challenger wins

## Rollback
Automatic rollback triggered if:
- Error rate > 1%
- Latency p99 > 500ms
- Drift score > 0.25
