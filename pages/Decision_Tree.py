import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


@st.cache_data
def load_dataset():
    return pd.read_csv("datasets/StrockDataset.csv")


df = load_dataset()
dataset_name = "Dataset về đột quỵ"
target_col = "stroke"
selected_features = [c for c in df.columns if c != target_col]
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

tab_data, tab_dt = st.tabs([
    "Dataset", "Decision Tree"
])

with tab_data:
    st.subheader("Xem Dataset")
    st.write(f"**{dataset_name}** — {df.shape[0]} dòng, {df.shape[1]} cột.")
    st.dataframe(df, use_container_width=True)

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), selected_features)]
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