import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from .preprocessing import build_preprocessor,build_models

def train_and_evaluate(df,mode,test_size=0.2):
    target="Waste_Type" if mode=="Waste Classification" else "fee"
    y=df[target].astype(str)
    # Avoid target leakage and identifier-like fields in Kaggle data.
    if mode=="Waste Classification":
        features=["Weight_kg","Moisture_Percent","Density","Organic_Percent","Recyclable"]
    else:
        excluded={target,"d_fee","istat","name","sample"}
        features=[c for c in df.columns if c not in excluded and df[c].notna().any()]
        # Drop high-cardinality textual identifiers; region/province remain useful geography.
        features=[c for c in features if not (df[c].dtype=="object" and df[c].nunique()> max(100,0.5*len(df)))]
    if not features: raise ValueError("No usable predictor columns remain after excluding target/leakage fields.")
    X=df[features].copy()
    counts=y.value_counts()
    if y.nunique()<2: raise ValueError(f"Target '{target}' must contain at least two classes.")
    if counts.min()<2: raise ValueError("Every target class needs at least 2 records for stratified evaluation.")
    n_classes=y.nunique()
    test_count=max(n_classes,int(round(len(y)*test_size)))
    if test_count>=len(y): raise ValueError("Not enough rows for a train/test split.")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_count,random_state=42,stratify=y)
    results,pipelines={},{}
    for name,model in build_models(len(X_train)).items():
        pipe=Pipeline([("preprocessor",build_preprocessor(X_train)),("model",model)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test)
        labels=sorted(y.unique())
        results[name]={
            "accuracy":accuracy_score(y_test,pred),
            "precision":precision_score(y_test,pred,average="weighted",zero_division=0),
            "recall":recall_score(y_test,pred,average="weighted",zero_division=0),
            "f1":f1_score(y_test,pred,average="weighted",zero_division=0),
            "confusion_matrix":confusion_matrix(y_test,pred,labels=labels),
            "labels":labels,
            "report":classification_report(y_test,pred,labels=labels,zero_division=0,output_dict=True),
            "train_rows":len(X_train),"test_rows":len(X_test)
        }
        pipelines[name]=pipe
    return results,pipelines,features
