# Kaggle dataset guide

This project recognizes the uploaded `public_data_waste_fee.csv` format through the columns `region`, `province`, `name`, and `fee`.

For this Kaggle file:
- Prediction target: `fee` (classes such as `PAYT` and `STANDARD`).
- Analytics fields: `region`, `province`, `name`, waste composition columns, `msw`, and `sor`.
- `d_fee` is excluded from model inputs because it directly encodes the fee target and would cause target leakage.
- Municipality identifiers (`istat`, `name`) and the sampling flag (`sample`) are excluded from model inputs.
- Missing predictor values are imputed inside each model pipeline.
- This file describes Italian municipalities; do not describe it as Raigad or Indian data.

To use it, start EcoSort AI and upload the Kaggle CSV in the sidebar. Select Model Comparison & Prediction to train and compare the three classifiers.
