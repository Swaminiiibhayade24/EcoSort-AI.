from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def build_preprocessor(X):
    numeric=list(X.select_dtypes(include=["number","bool"]).columns)
    categorical=[c for c in X.columns if c not in numeric]
    transformers=[]
    if numeric:
        transformers.append(("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric))
    if categorical:
        transformers.append(("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical))
    if not transformers:
        raise ValueError("No usable input features were found.")
    return ColumnTransformer(transformers,remainder="drop")

def build_models(train_rows):
    return {
        "Random Forest":RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42,n_jobs=-1),
        "SVM":SVC(C=2.0,kernel="rbf",probability=True,class_weight="balanced",random_state=42),
        "KNN":KNeighborsClassifier(n_neighbors=max(1,min(5,train_rows)),weights="distance")
    }
