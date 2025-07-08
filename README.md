# 📊 Data Analysis Assistant

A GenAI-powered Streamlit application that allows users to upload, merge, analyze, and query datasets with ease. Designed for data analysts, this tool supports interactive merging, automated visualizations, missing value reports, and natural language Q&A via LLMs.

## 🌐 Live App

👉 [Launch App](https://data-analysis-assistant-zsemfa9fdbd59b7wbhbr5r.streamlit.app/)

## 🎥 Demo

Watch the demo video here:  
👉 [Screen Recording](https://drive.google.com/file/d/1gFpZTYbOCGTpmquT3RuZ84PNs6tgBn00/view?usp=sharing)

## 🚀 Features

- Upload and preview multiple CSV files.
- Smart column matching and safe merge on common columns (inner, outer, left, right).
- Dynamic KPI metrics and visualizations (line, bar, box).
- Auto-generated missing value report.
- Natural language business questions powered by **DeepSeek LLM via Hugging Face (Fireworks)**.
- Built-in size checks to prevent crashing on large merges.

## 🧠 Powered by GenAI

Integrated with DeepSeek LLM using Hugging Face’s Inference API to:
- Interpret dataset samples.
- Answer user questions in plain English.
- Suggest relevant pandas code snippets.

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas, Seaborn, Matplotlib
- HuggingFace Hub (DeepSeek LLM)
- dotenv (for secure API key handling)

## 🧪 How to Run Locally

1. **Clone this repo**  
   ```bash
   git clone https://github.com/yourusername/data-analysis-assistant.git
   cd data-analysis-assistant
