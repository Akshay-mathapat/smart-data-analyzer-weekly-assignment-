import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

# -----------------------------
# Custom CSS (Glassmorphism UI)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Glass Card */
.glass {
    background: rgba(250, 250, 250, 0.12);
    border-radius: 20px;
    padding: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    margin-bottom: 20px;
    transition: 0.3s;
}
.glass:hover {
    transform: scale(1.02);
}

/* Fade Animation */
.fade-in {
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Center Titles */
h1, h2, h3 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1 class='fade-in'>📊 Smart Data Analyzer</h1>", unsafe_allow_html=True)

# -----------------------------
# Upload File
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    with st.spinner("Analyzing data..."):
        time.sleep(1)

    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # -----------------------------
    # Dashboard Metrics
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Rows", len(df))
    col2.metric("📊 Columns", len(df.columns))
    col3.metric("🔢 Numeric Features", len(numeric_cols))

    # -----------------------------
    # Dataset Preview + Stats
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
        st.subheader("📁 Dataset Preview")
        st.dataframe(df.head())
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
        st.subheader("📊 Statistical Summary")
        st.dataframe(df.describe())
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Probability Insights
    # -----------------------------
    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.subheader("🎲 Probability Insights")

    col = st.selectbox("Select Column", numeric_cols)

    min_val = float(df[col].min())
    max_val = float(df[col].max())

    range_vals = st.slider("Select Range", min_val, max_val, (min_val, max_val))

    prob = len(df[(df[col] >= range_vals[0]) & (df[col] <= range_vals[1])]) / len(df)

    st.success(f"📌 Probability that {col} is in range {range_vals}: {prob:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Vector Similarity
    # -----------------------------
    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.subheader("🔢 Vector Similarity")

    row1 = st.number_input("Row 1 Index", 0, len(df)-1, 0)
    row2 = st.number_input("Row 2 Index", 0, len(df)-1, 1)

    vec1 = df.loc[row1, numeric_cols].values
    vec2 = df.loc[row2, numeric_cols].values

    dot_product = np.dot(vec1, vec2)
    cos_sim = cosine_similarity([vec1], [vec2])[0][0]

    st.write(f"🔹 Dot Product: **{dot_product:.2f}**")
    st.write(f"🔹 Cosine Similarity: **{cos_sim:.2f}**")
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # Visualization (Modern Chart)
    # -----------------------------
    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.subheader("📈 Visual Comparison")

    chart_df = pd.DataFrame({
        "Features": numeric_cols,
        "Row 1": vec1,
        "Row 2": vec2
    })

    fig = px.bar(chart_df, x="Features", y=["Row 1", "Row 2"],
                 barmode="group", title="Row Comparison")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # AI Insights
    # -----------------------------
    st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)
    st.subheader("🤖 AI Insights")

    mean_val = df[col].mean()

    if mean_val > df[col].median():
        st.success("📈 Data is right-skewed (higher values more frequent)")
    else:
        st.info("📉 Data is left-skewed (lower values more frequent)")

    st.write(f"📌 Average of {col}: {mean_val:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)