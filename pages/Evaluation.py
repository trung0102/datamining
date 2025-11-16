from pages.Model import data_preprocessing, load_dataset

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np


df = load_dataset()
st.sidebar.markdown("---")
target_col = "Diabetes_binary"
selected_features = [col for col in df.columns if col != target_col]
st.sidebar.write("**Cột đặc trưng (features):**")
selected_features = st.sidebar.multiselect("Chọn features", options=[c for c in df.columns if c != target_col], default=selected_features)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

tab_miss, tab_tq, tab_ps = st.tabs([
    "Kiểm tra giá trị thiếu", "Kiểm tra tương quan", "Kiểm tra phương sai"
])

with tab_miss:
    st.subheader("Kiểm tra giá trị thiếu")
    missing_values = df.isnull().sum()
    missing_percent = (df.isnull().sum() / df.shape[0]) * 100
    missing_df = pd.DataFrame({
        'Cột': missing_values.index,
        'Số giá trị thiếu': missing_values.values,
        'Tỷ lệ thiếu (%)': missing_percent.values
    })
    st.dataframe(missing_df)
    if missing_df['Số giá trị thiếu'].sum() == 0:
        st.success("Không có giá trị thiếu trong dataset.")
    else:
        st.success("Gợi ý: Cân nhắc xử lý các cột có tỷ lệ thiếu cao (>50%) bằng cách điền giá trị hoặc loại bỏ.")

with tab_tq:
    st.subheader("Ma trận tương quan")
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    
    st.write("Các cặp cột có tương quan cao (|corr| > 0.8):")
    high_corr = np.where(np.abs(corr_matrix) > 0.8)
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                       for i, j in zip(*high_corr) if i != j and i < j]
    if high_corr_pairs:
        for col1, col2, corr_value in high_corr_pairs:
            st.write(f"{col1} và {col2}: {corr_value:.2f}")
    else:
        st.write("Không có cặp cột nào có tương quan cao.")

with tab_ps:
    st.subheader("Phương sai của các cột")
    variance = df.var()
    variance_df = pd.DataFrame({
        'Cột': variance.index,
        'Phương sai': variance.values
    })
    st.dataframe(variance_df)
    low_variance_cols = variance_df[variance_df['Phương sai'] < 0.01]['Cột'].tolist()
    if low_variance_cols:
        st.write(f"Các cột có phương sai thấp (< 0.01): {low_variance_cols}")
        st.write("Gợi ý: Cân nhắc loại bỏ các cột này vì chúng có thể không cung cấp nhiều thông tin.")
    else:
        st.write("Không có cột nào có phương sai quá thấp.")
