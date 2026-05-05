
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

MODEL_DIR = "./indic_bert_bengali_sentiment"
st.set_page_config(page_title="Bengali Sentiment Analyzer", layout="wide")
st.title("Bengali Review Sentiment Analyzer")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, top_k=None)

clf = load_model()

def predict_one(text):
    scores = clf(text)[0]
    best = max(scores, key=lambda x: x["score"])
    return best["label"], float(best["score"])

st.subheader("Single Review")
text_input = st.text_area("Paste one Bengali review", "")
if st.button("Predict"):
    if text_input.strip():
        lbl, conf = predict_one(text_input.strip())
        st.success(f"Label: {lbl} | Confidence: {conf:.2f}")
    else:
        st.warning("Please enter a review.")

st.divider()
st.subheader("Bulk CSV Analysis")
st.write("Upload a CSV with a text column (e.g., review).")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.write("Columns:", list(df.columns))

    text_col = st.selectbox("Select text column", df.columns)

    if st.button("Run Bulk Prediction"):
        texts = df[text_col].fillna("").astype(str).tolist()
        labels, confs = [], []
        for t in texts:
            lbl, c = predict_one(t)
            labels.append(lbl)
            confs.append(c)

        out = df.copy()
        out["pred_label"] = labels
        out["confidence"] = confs

        st.dataframe(out.head(20))

        st.subheader("Sentiment Distribution")
        counts = out["pred_label"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        ax.set_title("Predicted Sentiment")
        st.pyplot(fig)

        st.subheader("Top Themes (Simple Keywords)")
        vec = CountVectorizer(max_features=20, ngram_range=(1, 2), stop_words=None)
        X = vec.fit_transform(out[text_col].fillna("").astype(str))
        vocab = vec.get_feature_names_out()
        freqs = X.sum(axis=0).A1
        themes = pd.DataFrame({"theme": vocab, "count": freqs}).sort_values("count", ascending=False)
        st.dataframe(themes.head(10))

        st.download_button(
            "Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="bengali_sentiment_predictions.csv",
            mime="text/csv",
        )
