import pandas as pd

WASTE_REQUIRED = ["Date","Area","Waste_Type","Weight_kg","Moisture_Percent","Density","Organic_Percent","Recyclable"]
KAGGLE_REQUIRED = ["region","province","name","fee"]

def detect_mode(df):
    cols=set(df.columns)
    if set(WASTE_REQUIRED).issubset(cols):
        return "Waste Classification"
    if set(KAGGLE_REQUIRED).issubset(cols):
        return "Kaggle Municipal"
    raise ValueError(
        "Unsupported CSV format. Expected either the EcoSort columns "
        f"({', '.join(WASTE_REQUIRED)}) or Kaggle municipal columns "
        f"({', '.join(KAGGLE_REQUIRED)}). Your file has: {', '.join(map(str, df.columns[:15]))}"
    )

def load_csv(uploaded_file):
    try:
        uploaded_file.seek(0)
        raw=pd.read_csv(uploaded_file,encoding="utf-8-sig",low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    raw.columns=[str(c).replace("\\ufeff","").strip() for c in raw.columns]
    mode=detect_mode(raw)
    if mode=="Waste Classification":
        df=raw[WASTE_REQUIRED].copy()
        df["Date"]=pd.to_datetime(df["Date"],errors="coerce",dayfirst=True)
        for c in ["Weight_kg","Moisture_Percent","Density","Organic_Percent"]:
            df[c]=pd.to_numeric(df[c],errors="coerce")
        df["Area"]=df["Area"].astype("string").str.strip().fillna("Unknown")
        df["Waste_Type"]=df["Waste_Type"].astype("string").str.strip()
        rec=df["Recyclable"].astype("string").str.strip().str.lower()
        mapping={"yes":"Yes","y":"Yes","1":"Yes","true":"Yes","no":"No","n":"No","0":"No","false":"No"}
        df["Recyclable"]=rec.map(mapping)
        before=len(df)
        df=df.dropna(subset=["Waste_Type","Weight_kg","Moisture_Percent","Density","Organic_Percent","Recyclable"])
        df=df[(df.Weight_kg>=0)&(df.Density>=0)&df.Moisture_Percent.between(0,100)&df.Organic_Percent.between(0,100)]
        df["Waste_Type"]=df["Waste_Type"].astype(str)
        df["Area"]=df["Area"].astype(str)
        df["Recyclable"]=df["Recyclable"].astype(str)
        if df.empty: raise ValueError("No valid waste records remain after cleaning.")
    else:
        df=raw.copy()
        # Target must exist; keep missing feature values for model imputers.
        df["fee"]=df["fee"].astype("string").str.strip()
        df=df.dropna(subset=["fee"])
        df=df[df["fee"]!=""]
        # Remove completely empty columns; convert numeric-like object columns.
        df=df.dropna(axis=1,how="all")
        for c in df.columns:
            if c=="fee": continue
            if df[c].dtype=="object":
                converted=pd.to_numeric(df[c],errors="coerce")
                if converted.notna().mean()>=0.8:
                    df[c]=converted
                else:
                    df[c]=df[c].astype("string").fillna("Unknown").astype(str)
        if df["fee"].nunique()<2: raise ValueError("Kaggle fee target must contain at least two classes.")
        if len(df)<10: raise ValueError("Kaggle dataset has too few labelled records.")
    df=df.reset_index(drop=True)
    df.attrs["rows_removed"]=len(raw)-len(df)
    return df,mode

def dataset_quality(df):
    return {"rows":len(df),"columns":len(df.columns),"missing":int(df.isna().sum().sum()),"duplicates":int(df.duplicated().sum())}
