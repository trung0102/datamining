import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


@st.cache_data
def load_dataset():
    return pd.read_csv("datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

def data_preprocessing(df: pd.DataFrame):
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    target_col = "Diabetes_binary"
    selected_features = [col for col in df.columns if col != target_col]

    categorical_cols = [ "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
                            "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", 
                            "NoDocbcCost", "GenHlth", "DiffWalk", "Sex", "Education", "Income", "Age"]
    numerical_cols = ["BMI", "MentHlth", "PhysHlth"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), numerical_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", drop="first"))
            ]), categorical_cols)
        ]
    )
    return df, target_col, selected_features, preprocessor


df = load_dataset()
dataset_name = "Dataset về tiểu đường"

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Tỷ lệ test", 0.1, 0.5, 0.3, 0.05)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

tab_data, tab_dt = st.tabs([
    "Dataset", "Decision Tree"
])

with tab_data:
    st.subheader("Xem Dataset")
    st.write(f"**{dataset_name}** — {df.shape[0]} dòng, {df.shape[1]} cột.")
    st.dataframe(df, use_container_width=True)

df, target_col, selected_features, preprocessor = data_preprocessing(df)
X = df[selected_features].copy()
y = df[target_col].copy()
stratify_opt = None
if y.nunique() > 1 and y.value_counts().min() >= 2:
    stratify_opt = y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify_opt
)

with tab_dt:
    st.subheader("Decision Tree — Train/Test")
    col1, col2, col3 = st.columns(3)
    with col1:
        criterion = st.selectbox("criterion", ["entropy", "gini"], index=0, key="dt_criterion")
    with col2:
        max_depth = st.select_slider("max_depth", options=[None] + list(range(2, 21)), value=3, key="dt_max_depth")
    with col3:
        min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 1, 1, key="dt_min_leaf")

    dt = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    pipe_dt = Pipeline([("prep", preprocessor), ("clf", dt)])

    if st.button("Train Decision Tree"):
        pipe_dt.fit(X_train, y_train)
        y_pred = pipe_dt.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"**Accuracy (test)**: {acc:.4f}")
        st.markdown("**Classification report:**")
        st.code(classification_report(y_test, y_pred, zero_division=0), language="text")


