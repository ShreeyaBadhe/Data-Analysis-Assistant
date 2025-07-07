import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

import os
from huggingface_hub import InferenceClient
import streamlit as st
# Load API key

HF_API_KEY = st.secrets["HF_API_KEY"]
if not HF_API_KEY:
    st.error("❌ Hugging Face API key not found. Set HF_API_KEY in your .env file.")
    st.stop()

# LLM Setup
client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-0528",
    provider="fireworks-ai",
    api_key=HF_API_KEY
)

st.set_page_config(page_title="Dataset Merger & Analyzer", layout="wide")
st.title("📊 Dataset Merger & Analyzer")

uploaded_files = st.file_uploader("Upload 2 or more CSV files", type="csv", accept_multiple_files=True)
dataframes = []

# Merge block
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        dataframes.append(df)
        st.subheader(f"📄 Preview: {file.name}")
        st.dataframe(df.head())

    common_cols = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_cols &= set(df.columns)

    if common_cols:
        selected_cols = st.multiselect("🧩 Select column(s) to merge on", sorted(common_cols), default=list(common_cols)[:1])
        merge_type = st.selectbox("🔀 Merge type", ["inner", "outer", "left", "right"])

        total_rows = sum(len(df) for df in dataframes)
        estimated_merged_rows = total_rows ** 0.5 * 1.5

        if estimated_merged_rows > 1_000_000:
            st.warning(f"⚠️ Merge skipped: Estimated result size too large ({int(estimated_merged_rows):,} rows).")
        elif st.button("🔧 Merge Datasets"):
            try:
                merged_df = reduce(lambda l, r: pd.merge(l, r, on=selected_cols, how=merge_type), dataframes)
                st.session_state["merged_df"] = merged_df  # store for future interactions
                st.success(f"✅ Merge successful! Rows: {len(merged_df)}")
                csv = merged_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Merged CSV", csv, "merged.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ Merge failed: {e}")
    else:
        st.warning("⚠️ No common columns to merge on.")

# Analysis section
if "merged_df" in st.session_state:
    merged_df = st.session_state["merged_df"]

    st.subheader("🧬 Merged Dataset")
    st.dataframe(merged_df.head())

    st.subheader("📉 Missing Value Report")
    missing = merged_df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    missing["% Missing"] = (missing["Missing Values"] / len(merged_df)) * 100
    st.dataframe(missing)

    st.subheader("💬 Ask a Business Question")
    with st.form("question_form"):
        question = st.text_area("E.g. What is the average revenue per product?")
        submit = st.form_submit_button("🤖 Ask LLM")

    def ask_llm(df, question):
        sample = df.sample(min(3, len(df)), random_state=42).iloc[:, :5].to_markdown(index=False)
        prompt = f"""
You are a data expert. Here's a sample of the dataset:

{sample}

Question: {question}

Provide a clear answer. Include pandas code if helpful.
"""
        res = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        return res.choices[0].message.content

    if submit and question:
        with st.spinner("🤖 Thinking..."):
            result = ask_llm(merged_df, question)
            st.markdown("### 🧠 LLM Response:")
            st.code(result)

    # KPI
    numeric_cols = merged_df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        with st.expander("📌 KPI Summary"):
            metric = st.selectbox("Choose numeric column", numeric_cols)
            st.write(f"**Total {metric}:**", merged_df[metric].sum())
            st.write(f"**Average {metric}:**", merged_df[metric].mean())

    # Visualization
    st.subheader("📈 Visualize Metrics")
    datetime_cols = merged_df.select_dtypes(include='datetime').columns.tolist()
    all_cols = merged_df.columns.tolist()

    col_x = st.selectbox("X-axis (Date/Time preferred):", options=datetime_cols or all_cols)
    col_y = st.selectbox("Y-axis (Numeric):", options=numeric_cols)
    chart_type = st.selectbox("Chart type", ["Line", "Bar", "Box"])

    if st.button("📊 Generate Chart"):
        fig, ax = plt.subplots()
        try:
            if chart_type == "Line":
                sns.lineplot(data=merged_df, x=col_x, y=col_y, ax=ax)
            elif chart_type == "Bar":
                sns.barplot(data=merged_df, x=col_x, y=col_y, ax=ax)
            elif chart_type == "Box":
                sns.boxplot(data=merged_df, x=col_x, y=col_y, ax=ax)
            ax.set_title(f"{chart_type} Chart of {col_y} by {col_x}")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Chart Error: {e}")
