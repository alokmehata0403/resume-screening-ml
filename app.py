import os
import re
import pickle
import pandas as pd
import pdfplumber
import streamlit as st

# ---------- Paths Setup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------- Load models (Cached) ----------
@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "clf.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return tfidf, model, encoder

tfidf, model, encoder = load_models()

# ---------- Clean text ----------
def clean_resume(text):
    text = re.sub(r'http\S+', ' ', str(text))
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# ---------- Page config ----------
st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.title("AI Resume Screening System")
st.write("Upload resume (PDF) to automatically classify the candidate into the most relevant job role.")

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + " "

    # Edge Case: Scanned image or blank PDF
    if not text.strip():
        st.error("⚠️ Is PDF se text extract nahi ho paya. Kripya digital/searchable text wali PDF upload karein (scanned image nahi).")
    else:
        st.subheader("Extracted Resume Text (Preview)")
        st.text_area("Extracted Content", text[:2000], height=150)

        if st.button("Analyze Resume"):
            cleaned = clean_resume(text)
            vector = tfidf.transform([cleaned])

            # Edge Case: Completely unrelated text / no vocabulary matches
            if vector.nnz == 0:
                st.warning("⚠️ Resume me dataset ke model se match hone wale relevant skills/keywords nahi mile.")
            else:
                prediction = model.predict(vector)
                probabilities = model.predict_proba(vector)[0]

                category = encoder.inverse_transform(prediction)[0]
                match_score = max(probabilities) * 100

                # ---------- Result section ----------
                col1, col2 = st.columns(2)

                with col1:
                    st.success(f"**Predicted Role:** {category}")

                with col2:
                    st.info(f"**Match Confidence:** {match_score:.2f}%")

                # ---------- Confidence chart ----------
                proba_df = pd.DataFrame({
                    "Category": encoder.classes_,
                    "Confidence": probabilities
                }).sort_values(by="Confidence", ascending=False)

                st.subheader("Top Prediction Confidence")
                st.bar_chart(
                    proba_df.set_index("Category").head(5),
                    use_container_width=True
                )
