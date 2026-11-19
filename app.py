import streamlit as st
import pandas as pd
import plotly.express as px
from modules.dataset_loader import load_csv, dataset_quality, detect_mode
from modules.evaluation import train_and_evaluate

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")
st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#061b14,#0b3023);color:#e8f4ed}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#071d15,#10412e)}
.hero {padding:28px;border-radius:18px;background:linear-gradient(110deg,#0d4933,#126447);border:1px solid #237a58;margin-bottom:18px}
.hero h1 {color:#e9fff2}
div[data-testid="stMetric"] {background:#103b2b;padding:14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "trained" not in st.session_state:
    st.session_state.trained = None
if "dataset_mode" not in st.session_state:
    st.session_state.dataset_mode = None

with st.sidebar:
    st.markdown("# ♻️ EcoSort AI")
    st.caption("SMART WASTE • SMARTER CITIES")
    page = st.radio("NAVIGATION", [
        "🏠 Dashboard", "📊 Dataset Explorer", "♻️ Waste Analytics",
        "🌳 Random Forest", "🧠 SVM", "🔍 KNN", "⚖️ Model Comparison & Prediction"
    ])
    st.markdown("---")
    st.markdown("### 📤 Upload CSV")
    st.caption("Upload your original waste CSV or supported Kaggle municipal CSV.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="sidebar_upload", label_visibility="collapsed")
    if uploaded is not None:
        signature = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_signature") != signature:
            try:
                df, mode = load_csv(uploaded)
                st.session_state.dataset = df
                st.session_state.dataset_mode = mode
                st.session_state.dataset_source = uploaded.name
                st.session_state.trained = None
                st.session_state.upload_signature = signature
                st.success(f"Loaded {len(df):,} rows ({mode}).")
            except Exception as e:
                st.session_state.dataset = None
                st.session_state.dataset_mode = None
                st.session_state.trained = None
                st.session_state.upload_signature = None
                st.error(f"Upload failed: {e}")
    if st.session_state.dataset is not None:
        st.markdown("---")
        st.markdown("### CURRENT DATASET")
        st.success(f"{len(st.session_state.dataset):,} records")
        st.caption(f"Mode: {st.session_state.dataset_mode}")
        st.caption(f"File: {st.session_state.dataset_source}")
        if st.button("Remove dataset", use_container_width=True):
            for k in ["dataset","dataset_mode","dataset_source","trained","upload_signature"]:
                st.session_state[k] = None
            st.rerun()

if st.session_state.dataset is None:
    hero("Welcome to EcoSort AI", "Upload a waste dataset to begin.")
    st.info("Supported formats: (1) EcoSort waste-classification CSV and (2) Kaggle municipal waste-fee CSV with region, province, name and fee columns.")
    st.markdown("### What happens after upload?")
    st.write("EcoSort identifies the dataset type, validates columns, prepares analytics and trains Random Forest, SVM and KNN on an appropriate labelled target.")
    st.stop()

df = st.session_state.dataset
mode = st.session_state.dataset_mode
target = "Waste_Type" if mode == "Waste Classification" else "fee"

if page == "🏠 Dashboard":
    hero("Waste intelligence, made actionable.", f"{st.session_state.dataset_source} · {mode}")
    if mode == "Waste Classification":
        total = df["Weight_kg"].sum()
        c1,c2,c3,c4=st.columns(4)
        c1.metric("TOTAL WASTE",f"{total:,.1f} kg")
        c2.metric("RECORDS",f"{len(df):,}")
        c3.metric("WASTE CATEGORIES",df["Waste_Type"].nunique())
        c4.metric("AREAS",df["Area"].nunique())
        left,right=st.columns(2)
        with left:
            cat=df.groupby("Waste_Type",as_index=False)["Weight_kg"].sum()
            st.plotly_chart(px.pie(cat,names="Waste_Type",values="Weight_kg",hole=.45,template="plotly_dark"),use_container_width=True)
        with right:
            area=df.groupby("Area",as_index=False)["Weight_kg"].sum().sort_values("Weight_kg",ascending=False).head(20)
            st.plotly_chart(px.bar(area,x="Area",y="Weight_kg",template="plotly_dark"),use_container_width=True)
    else:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("MUNICIPAL RECORDS",f"{len(df):,}")
        c2.metric("REGIONS",df["region"].nunique())
        c3.metric("PROVINCES",df["province"].nunique())
        c4.metric("FEE TYPES",df["fee"].nunique())
        left,right=st.columns(2)
        with left:
            st.plotly_chart(px.histogram(df,x="fee",color="fee",title="Municipal fee system distribution",template="plotly_dark"),use_container_width=True)
        with right:
            agg=df.groupby("region",as_index=False)["msw"].sum().sort_values("msw",ascending=False).head(15)
            st.plotly_chart(px.bar(agg,x="region",y="msw",title="Municipal solid waste by region",template="plotly_dark"),use_container_width=True)
    st.caption("All statistics are calculated from the uploaded file.")

elif page == "📊 Dataset Explorer":
    hero("Dataset Explorer","Explore the actual uploaded records.")
    q=dataset_quality(df)
    a,b,c,d=st.columns(4)
    a.metric("ROWS",q["rows"]); b.metric("COLUMNS",q["columns"]); c.metric("MISSING CELLS",q["missing"]); d.metric("DUPLICATES",q["duplicates"])
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.subheader("Column summary")
    st.dataframe(df.describe(include="all").transpose().fillna("—"),use_container_width=True)
    st.download_button("Download processed CSV",df.to_csv(index=False).encode("utf-8"),"ecosort_processed.csv","text/csv")

elif page == "♻️ Waste Analytics":
    hero("Waste Analytics","Explore composition, geography and trends.")
    if mode == "Waste Classification":
        a,b=st.columns(2)
        with a:
            cat=df.groupby("Waste_Type",as_index=False)["Weight_kg"].sum()
            st.plotly_chart(px.bar(cat,x="Waste_Type",y="Weight_kg",template="plotly_dark"),use_container_width=True)
        with b:
            area=df.groupby("Area",as_index=False)["Weight_kg"].sum().sort_values("Weight_kg",ascending=False)
            st.plotly_chart(px.bar(area,x="Area",y="Weight_kg",template="plotly_dark"),use_container_width=True)
        rec=df.groupby("Recyclable",as_index=False)["Weight_kg"].sum()
        st.plotly_chart(px.pie(rec,names="Recyclable",values="Weight_kg",hole=.4,template="plotly_dark"),use_container_width=True)
        if df["Date"].notna().any():
            trend=df.dropna(subset=["Date"]).set_index("Date")["Weight_kg"].resample("MS").sum().reset_index()
            st.plotly_chart(px.line(trend,x="Date",y="Weight_kg",markers=True,template="plotly_dark"),use_container_width=True)
    else:
        metric=st.selectbox("Municipal waste metric",["msw","msw_so","msw_un","organic","paper","glass","plastic","metal","tc","sor"])
        geo=st.selectbox("Group by",["region","province"])
        agg=df.groupby(geo,as_index=False)[metric].sum(min_count=1).sort_values(metric,ascending=False).head(30)
        st.plotly_chart(px.bar(agg,x=geo,y=metric,color=geo,template="plotly_dark",title=f"{metric} by {geo}"),use_container_width=True)
        st.subheader("Explore municipalities")
        regions=["All"]+sorted(df["region"].dropna().astype(str).unique())
        chosen=st.selectbox("Region filter",regions)
        subset=df if chosen=="All" else df[df.region.astype(str)==chosen]
        st.dataframe(subset[["region","province","name","fee","pop","msw","sor"]].head(500),use_container_width=True,hide_index=True)

else:
    model_page={"🌳 Random Forest":"Random Forest","🧠 SVM":"SVM","🔍 KNN":"KNN"}.get(page)
    hero("Model Comparison & Prediction" if model_page is None else f"{model_page} Model Lab",
         f"Classification target: {target}. Models use the uploaded dataset only.")
    if st.session_state.trained is None:
        with st.spinner("Training Random Forest, SVM and KNN..."):
            try:
                st.session_state.trained=train_and_evaluate(df,mode)
            except Exception as e:
                st.error(f"Model training could not complete: {e}")
                st.info("Check target labels, class counts and available feature columns.")
                st.stop()
    results,models,features=st.session_state.trained
    if model_page:
        r=results[model_page]
        a,b,c,d=st.columns(4)
        a.metric("ACCURACY",f"{r['accuracy']:.1%}"); b.metric("PRECISION",f"{r['precision']:.1%}")
        c.metric("RECALL",f"{r['recall']:.1%}"); d.metric("F1 SCORE",f"{r['f1']:.1%}")
        st.caption(f"Target: {target} · Features: {', '.join(features)} · Train: {r['train_rows']:,} · Test: {r['test_rows']:,}")
        cm=pd.DataFrame(r["confusion_matrix"],index=r["labels"],columns=r["labels"])
        st.plotly_chart(px.imshow(cm,text_auto=True,aspect="auto",template="plotly_dark",title="Confusion matrix"),use_container_width=True)
        st.dataframe(pd.DataFrame(r["report"]).transpose().round(3),use_container_width=True)
        if model_page=="Random Forest" and hasattr(models[model_page].named_steps["model"],"feature_importances_"):
            names=models[model_page].named_steps["preprocessor"].get_feature_names_out()
            imp=pd.DataFrame({"Feature":names,"Importance":models[model_page].named_steps["model"].feature_importances_}).sort_values("Importance").tail(20)
            st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h",template="plotly_dark"),use_container_width=True)
    else:
        comp=pd.DataFrame([{"Algorithm":n,"Accuracy":r["accuracy"],"Precision":r["precision"],"Recall":r["recall"],"F1-score":r["f1"]} for n,r in results.items()])
        st.dataframe(comp.style.format({c:"{:.1%}" for c in ["Accuracy","Precision","Recall","F1-score"]}),use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(comp.melt(id_vars="Algorithm",var_name="Metric",value_name="Score"),x="Algorithm",y="Score",color="Metric",barmode="group",range_y=[0,1],template="plotly_dark"),use_container_width=True)
        st.success(f"Best model by weighted F1-score: {comp.sort_values('F1-score',ascending=False).iloc[0]['Algorithm']}")
        st.subheader("Predict using a new record")
        st.caption("The input starts with typical values from the uploaded dataset. Edit the values before predicting.")
        base=df[features].median(numeric_only=True).to_dict()
        row={}
        for col in features:
            if pd.api.types.is_numeric_dtype(df[col]):
                row[col]=base.get(col,0)
            else:
                modevals=df[col].dropna().mode()
                row[col]=str(modevals.iloc[0]) if len(modevals) else "Unknown"
        edited=st.data_editor(pd.DataFrame([row]),use_container_width=True,num_rows="fixed")
        chosen=st.selectbox("Prediction model",list(models.keys()))
        if st.button("Predict target"):
            try:
                pred=models[chosen].predict(edited[features])[0]
                st.success(f"Predicted {target}: {pred}")
                if hasattr(models[chosen],"predict_proba"):
                    probs=models[chosen].predict_proba(edited[features])[0]
                    st.dataframe(pd.DataFrame({"Class":models[chosen].classes_,"Probability":probs}).sort_values("Probability",ascending=False),use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
