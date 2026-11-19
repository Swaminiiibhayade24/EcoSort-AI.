# EcoSort AI — Dual Dataset Edition

EcoSort AI supports two CSV workflows:

1. **Waste Classification** — requires `Date, Area, Waste_Type, Weight_kg, Moisture_Percent, Density, Organic_Percent, Recyclable`; predicts `Waste_Type`.
2. **Kaggle Municipal Waste** — recognizes the uploaded Kaggle municipal dataset with `region, province, name, fee`; analyzes regional/municipal waste and predicts `fee` (for example PAYT vs STANDARD).

The Kaggle workflow uses the fee column as the target. It excludes `d_fee` because it is a direct duplicate/encoding of the target, and excludes municipality name/ISTAT/sample identifier fields from predictors. Missing predictor values are imputed inside each training pipeline. The dataset is split into stratified training and testing sets.

## Run on Windows

```bat
cd /d "C:\Users\Swamini\OneDrive\codes\EcoSort_AI"
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Or double-click `RUN_ECOSORT.bat`.

All analytics and model training use the uploaded file. The app does not silently load demo data. The Kaggle dataset is Italian municipal data; it is not Raigad/Maharashtra data. Use a separately sourced and verified Raigad dataset for local claims.
