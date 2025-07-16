# app.py

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Dynamic User Segmentation", layout="wide")
st.title("🧠 Dynamic User Segmentation App (No Pre-trained Model)")

uploaded_file = st.file_uploader("📁 Upload any CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🧾 Raw Data Preview")
    st.dataframe(df.head())

    # Clean columns
    df.columns = df.columns.str.strip()

    # Optional: gender encoding
    if 'Gender' in df.columns:
        gender_check = st.checkbox("Encode Gender column?")
        if gender_check:
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # Allow user to select features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_features = st.multiselect("Select features for clustering (min 2)", numeric_cols)

    if len(selected_features) >= 2:
        try:
            X = df[selected_features]

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # KMeans Clustering
            k = st.slider("Number of Clusters", min_value=2, max_value=6, value=4)
            model = KMeans(n_clusters=k, random_state=42)
            clusters = model.fit_predict(X_scaled)
            df['Segment'] = clusters

            st.success("✅ Segmentation complete!")

            st.subheader("📋 Segmented Data")
            st.dataframe(df)

            # Plot
            x_axis = st.selectbox("X-axis", selected_features)
            y_axis = st.selectbox("Y-axis", selected_features, index=1)

            fig = px.scatter(df, x=x_axis, y=y_axis, color="Segment", title="Cluster Visualization", hover_data=df.columns)
            st.plotly_chart(fig, use_container_width=True)

            st.download_button("⬇️ Download Segmented Data", data=df.to_csv(index=False), file_name="segmented_users.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error while clustering: {e}")
    else:
        st.warning("📌 Select at least 2 numeric features for clustering.")
else:
    st.info("👆 Upload any CSV file to begin.")
