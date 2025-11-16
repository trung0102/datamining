import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


@st.cache_data
def load_dataset():
    return pd.read_csv("datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    # return pd.read_csv("datasets/diabetes_binary_health_indicators_BRFSS2015.csv")

def data_preprocessing(df: pd.DataFrame, selected_features):
    df = df.copy()

    binary_cols = [
        "CholCheck", "Smoker", "Stroke",
        "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "DiffWalk", "Sex"
    ]
    ordinal_cols = ["GenHlth", "Education", "Income"]
    numerical_cols = ["BMI", "MentHlth", "PhysHlth"]

    binary_cols = [col for col in binary_cols if col in selected_features]
    ordinal_cols = [col for col in ordinal_cols if col in selected_features]
    numerical_cols = [col for col in numerical_cols if col in selected_features]

    df["BMI_Category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    df['Age_binned'] = pd.cut(
            df['Age'], bins=[0, 3, 7, 11, 13], labels=[0, 1, 2, 3]
        ).astype(int)
    
    df['MentHlth_Category'] = pd.cut(
        df['MentHlth'], bins=[-1, 0, 5, 15, 30], labels=[0, 1, 2, 3]
    ).astype(int)

    
    df['PhysHlth_Category'] = pd.cut(
        df['PhysHlth'], bins=[-1, 0, 5, 15, 30], labels=[0, 1, 2, 3]
    ).astype(int)

    
    df["CardioRisk"] = df[["HighBP", "HighChol", "HeartDiseaseorAttack"]].sum(axis=1)

    
    df['BMI_Age_Interaction'] = df['BMI_Category'] * df['Age_binned']

    
    binary_cols += ['BMI_Category', 'MentHlth_Category', 'PhysHlth_Category']
    numerical_cols = ['CardioRisk', 'BMI_Age_Interaction'] 
    ordinal_cols += ['Age_binned']
    selected_features = binary_cols + ordinal_cols + numerical_cols

    # df = df[(df['BMI'] >= 12) & (df['BMI'] <= 60)]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median"))
            ]), numerical_cols),
            ("binary", "passthrough", binary_cols),
            ("ordinal", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent"))
            ]), ordinal_cols)
        ]
    )
    return df, selected_features, preprocessor

df = load_dataset()
df.drop_duplicates(inplace=True)
dataset_name = "Dataset về tiểu đường"

st.sidebar.markdown("---")
test_size = st.sidebar.slider("Tỷ lệ test", 0.1, 0.5, 0.3, 0.05)
target_col = "Diabetes_binary"
selected_features = [col for col in df.columns if col != target_col]
st.sidebar.write("**Cột đặc trưng (features):**")
selected_features = st.sidebar.multiselect("Chọn features", options=[c for c in df.columns if c != target_col], default=selected_features)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

df, selected_features, preprocessor = data_preprocessing(df, selected_features)
X = df[selected_features].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

tab_data, tab_dt, tab_rf, tab_cv, tab_gs = st.tabs([
    "Dataset", "Decision Tree", "Random Forest", "Cross Validation", "GridSearchCV"
])

with tab_data:
    st.subheader("Xem Dataset")
    st.write(f"**{dataset_name}** — {df.shape[0]} dòng, {df.shape[1]} cột.")
    st.dataframe(df, use_container_width=True)
    st.markdown("**Cột đã chọn:**")
    st.write({"target": target_col, "features": selected_features})

with tab_dt:
    st.subheader("Decision Tree — Train/Test")
    col1, col2, col3 = st.columns(3)
    with col1:
        criterion = st.selectbox("criterion", ["entropy", "gini"], index=0, key="dt_criterion")
    with col2:
        max_depth = st.select_slider("max_depth", options=[None] + list(range(2, 21)), value=8, key="dt_max_depth")
    with col3:
        min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 3, 1, key="dt_min_leaf")

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

        # Confusion matrix
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm = plt.figure(figsize=(6, 4.5))
        ax = fig_cm.add_subplot(111)
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix — Decision Tree")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

        # Feature importances + tree plot
        feat_names = pipe_dt.named_steps["prep"].get_feature_names_out()

        if hasattr(pipe_dt.named_steps["clf"], "feature_importances_"):
            importances = pipe_dt.named_steps["clf"].feature_importances_
            order = np.argsort(importances)[::-1]
            top_k = min(20, len(importances))

            fig_fi = plt.figure(figsize=(7, 5))
            ax = fig_fi.add_subplot(111)
            y_pos = np.arange(top_k)
            ax.barh(y_pos, importances[order][:top_k])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_names[order][:top_k])
            ax.invert_yaxis()
            ax.set_xlabel("Importance"); ax.set_title("Top Feature Importances — DT")
            fig_fi.tight_layout()
            st.pyplot(fig_fi)

        st.markdown("**Decision Tree plot:**")
        fig_tree = plt.figure(figsize=(12, 7))
        plot_tree(
            pipe_dt.named_steps["clf"],
            feature_names=feat_names,
            class_names=[str(c) for c in sorted(df[target_col].unique())],
            filled=False,
            rounded=True
        )
        plt.title("Decision Tree")
        plt.tight_layout()
        st.pyplot(fig_tree)


with tab_rf:
    st.subheader("Random Forest — Train/Test")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50, key="rf_n")
    with col2:
        max_depth_rf = st.select_slider("max_depth", options=[None] + list(range(2, 21)), value=8, key="rf_max")
    with col3:
        min_samples_leaf_rf = st.slider("min_samples_leaf", 1, 10, 3, 1, key="rf_min_leaf")
    with col4:
        max_features = st.selectbox("max_features", ["sqrt", "log2"], index=0, key="rf_mf")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth_rf,
        # min_samples_split=7,
        min_samples_leaf=min_samples_leaf_rf,
        max_features=max_features,
        random_state=random_state
    )
    pipe_rf = Pipeline([("prep", preprocessor), ("clf", rf)])

    if st.button("Train Random Forest", key="btn_rf"):
        pipe_rf.fit(X_train, y_train)
        y_pred = pipe_rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"**Accuracy (test)**: {acc:.4f}")
        # y_pred_train = pipe_rf.predict(X_train)
        # acc_train = accuracy_score(y_train, y_pred_train)
        # st.info(f"**Train Accuracy**: {acc_train:.4f}")
        st.markdown("**Classification report:**")
        st.code(classification_report(y_test, y_pred, zero_division=0), language="text")

        # Confusion matrix
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig_cm = plt.figure(figsize=(6, 4.5))
        ax = fig_cm.add_subplot(111)
        ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix — Random Forest")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

        # Feature importances
        feat_names = pipe_dt.named_steps["prep"].get_feature_names_out()
        if hasattr(pipe_rf.named_steps["clf"], "feature_importances_"):
            importances = pipe_rf.named_steps["clf"].feature_importances_
            order = np.argsort(importances)[::-1]
            top_k = min(20, len(importances))
            fig_fi = plt.figure(figsize=(7, 5))
            ax = fig_fi.add_subplot(111)
            y_pos = np.arange(top_k)
            ax.barh(y_pos, importances[order][:top_k])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feat_names[order][:top_k])
            ax.invert_yaxis()
            ax.set_xlabel("Importance"); ax.set_title("Top Feature Importances — RF")
            fig_fi.tight_layout()
            st.pyplot(fig_fi)

with tab_cv:
    st.subheader("Cross-Validation (StratifiedKFold)")
    
    class_counts = {label: int((y == label).sum()) for label in y.unique()}
    min_count = min(class_counts.values())
    default_folds = int(max(2, min(5, min_count)))
    n_splits = st.slider("n_splits (≤ lớp ít mẫu)", 2, 10, default_folds, 1)

    model_cv_name = st.selectbox("Mô hình", ["Decision Tree", "Random Forest"], index=1, key="cv_model_select")
    if model_cv_name == "Decision Tree":
        criterion_cv = st.selectbox("criterion", ["entropy", "gini"], index=0, key="cv_dt_crit")
        max_depth_cv = st.select_slider("max_depth", options=[None] + list(range(2, 21)), value=3, key="cv_dt_max")
        min_leaf_cv = st.slider("min_samples_leaf", 1, 10, 1, 1, key="cv_dt_leaf")
        model_cv = DecisionTreeClassifier(
            criterion=criterion_cv, max_depth=max_depth_cv, min_samples_leaf=min_leaf_cv, random_state=random_state
        )
    else:
        n_est_cv = st.slider("n_estimators", 50, 500, 200, 50, key="cv_rf_n")
        max_depth_cv = st.select_slider("max_depth", options=[None] + list(range(2, 21)), value=None, key="cv_rf_max")
        min_leaf_cv = st.slider("min_samples_leaf", 1, 10, 1, 1, key="cv_rf_leaf")
        max_feat_cv = st.selectbox("max_features", ["sqrt", "log2"], index=0, key="cv_rf_mf")
        model_cv = RandomForestClassifier(
            n_estimators=n_est_cv, max_depth=max_depth_cv, min_samples_leaf=min_leaf_cv,
            max_features=max_feat_cv, random_state=random_state
        )

    pipe_cv = Pipeline([("prep", preprocessor), ("clf", model_cv)])

    if st.button("Run Cross - Validation", key="btn_cv"):
        n_splits = int(max(2, min(n_splits, min_count)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipe_cv, X, y, cv=cv, scoring="accuracy")
        st.success(f"CV Accuracy: **{scores.mean():.4f} ± {scores.std():.4f}** (n={len(scores)})")

with tab_gs:
    st.subheader("GridSearchCV — so sánh cấu hình")
    gs_model_name = st.selectbox("Mô hình", ["Decision Tree", "Random Forest"], index=1, key="gs_model_select")

    if gs_model_name == "Decision Tree":
        st.info("Lưới thử cho Decision Tree")
        max_depth_list = st.multiselect("max_depth (chọn nhiều)", options=[None, 2, 3, 4, 5, 6, 8, 10], default=[None, 3, 5])
        min_leaf_list = st.multiselect("min_samples_leaf (chọn nhiều)", options=list(range(1,6)), default=[1,2,3])
        crit_list = st.multiselect("criterion", options=["entropy","gini"], default=["entropy","gini"])
        base = DecisionTreeClassifier(random_state=random_state)
        param_grid = {
            "clf__criterion": crit_list,
            "clf__max_depth": max_depth_list,
            "clf__min_samples_leaf": min_leaf_list,
        }
    else:
        st.info("Lưới thử cho Random Forest")
        n_list = st.multiselect("n_estimators", options=[100, 200, 400], default=[100,200])
        max_depth_list = st.multiselect("max_depth", options=[None, 3, 5, 8, 10], default=[None,3,5])
        min_leaf_list = st.multiselect("min_samples_leaf", options=list(range(1,6)), default=[1,2,3])
        mf_list = st.multiselect("max_features", options=["sqrt","log2"], default=["sqrt","log2"])
        base = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "clf__n_estimators": n_list,
            "clf__max_depth": max_depth_list,
            "clf__min_samples_leaf": min_leaf_list,
            "clf__max_features": mf_list,
        }

    pipe_gs = Pipeline([("prep", preprocessor), ("clf", base)])

    class_counts = {label: int((y == label).sum()) for label in y.unique()}
    min_count = min(class_counts.values())
    default_folds = int(max(2, min(5, min_count)))
    n_splits_gs = st.slider("n_splits (CV)", 2, 10, default_folds, 1, key="gs_folds")

    if st.button("Run GridSearchCV", key="btn_gs"):
        n_splits_gs = int(max(2, min(n_splits_gs, min_count)))
        cv = StratifiedKFold(n_splits=n_splits_gs, shuffle=True, random_state=random_state)
        gs = GridSearchCV(pipe_gs, param_grid=param_grid, scoring="recall", cv=cv, refit=True, return_train_score=False)
        gs.fit(X, y)

        res_sorted = pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False)
        st.success(f"Best params: **{gs.best_params_}**")
        st.write(f"Best CV recall: **{gs.best_score_:.4f}**")

        topk = min(10, len(res_sorted))
        st.markdown(f"**Top {topk} cấu hình (Mean CV Accuracy):**")
        
        cols_to_show = [c for c in res_sorted.columns if c.startswith("param_")] + ["mean_test_score"]
        st.dataframe(res_sorted[cols_to_show].head(topk), use_container_width=True)