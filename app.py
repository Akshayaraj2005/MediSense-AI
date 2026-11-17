
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# Page configuration
st.set_page_config(
    page_title="MediSense AI",
    page_icon="🏥",
    layout="wide"
)

# Load model
model = joblib.load("medisense_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Stopwords
stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"not", "no", "nor"}

# Text preprocessing
def preprocess_review(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Custom CSS
st.markdown("""
<style>
.stApp {
    background: #f4f8fb;
}

.main-title {
    color: #087e8b;
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    color: #526777;
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

[data-testid="stSidebar"] {
    background-color: #e6f3f5;
}

.stTextArea textarea {
    background-color: white;
    border: 2px solid #c7e2e5;
    border-radius: 12px;
    font-size: 16px;
}

.stButton button {
    background-color: #087e8b;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}

.stButton button:hover {
    background-color: #05636d;
    color: white;
    border: none;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

h1, h2, h3 {
    color: #174b5a;
}

</style>
""", unsafe_allow_html=True)


# Header



st.markdown("""
<div class="main-title">
    🏥 MediSense AI
</div>
<div class="subtitle">
    Healthcare Review Intelligence System
</div>
<div style="
    background: white;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #dcebed;
    margin-bottom: 25px;
">
    <h3 style="color:#087e8b; margin:0;">
        Welcome to MediSense AI
    </h3>
    <p style="color:#526777; margin:8px 0 0 0;">
        Smart Hospital Feedback Analysis Using Machine Learning & NLP
    </p>
</div>
""", unsafe_allow_html=True)



st.write("")
st.divider()

# Sidebar
with st.sidebar:
    st.title("🏥 MediSense AI")
    st.write("Healthcare Review Analysis")
    st.divider()
    st.subheader("About the Project")
    st.write(
        "MediSense AI uses Natural Language Processing "
        "and Machine Learning to analyze hospital reviews."
    )
    st.info("Algorithm: Linear Support Vector Machine")

# Main section


st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background:#e3f2fd; padding:20px;
    border-radius:15px; text-align:center;">
        <h4 style="color:#1769aa;">Total Reviews</h4>
        <h1 style="color:#1769aa;">977</h1>
        <p style="color:#526777;">Patient Feedback</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#e5f5e9; padding:20px;
    border-radius:15px; text-align:center;">
        <h4 style="color:#23864b;">Positive Reviews</h4>
        <h1 style="color:#23864b;">712</h1>
        <p style="color:#526777;">72.88% of reviews</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#fdeaea; padding:20px;
    border-radius:15px; text-align:center;">
        <h4 style="color:#c0392b;">Negative Reviews</h4>
        <h1 style="color:#c0392b;">265</h1>
        <p style="color:#526777;">27.12% of reviews</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")


st.markdown("""
<div style="
    background:white;
    padding:20px;
    border-radius:15px;
    border-left:5px solid #087e8b;
    margin-top:20px;
    margin-bottom:15px;
">
    <h3 style="color:#087e8b; margin:0;">
        📝 Patient Review Analysis
    </h3>
    <p style="color:#526777; margin:8px 0 0 0;">
        Enter patient feedback to analyze its sentiment using AI.
    </p>
</div>
""", unsafe_allow_html=True)


review = st.text_area(
    "Enter Patient Review",
    placeholder="Example: The doctors were helpful and the treatment was excellent...",
    height=150
)

if st.button("🔍 Analyze Review", use_container_width=True, key="analyze_review"):

    if review.strip() == "":
        st.warning("Please enter a hospital review.")

    else:
        cleaned_review = preprocess_review(review)
        review_tfidf = tfidf.transform([cleaned_review])
        prediction = model.predict(review_tfidf)

        st.divider()
        st.subheader("📊 Analysis Result")

        

        if prediction[0] == 1:
            st.markdown("""
            <div style="
                background:#e5f5e9;
                border-left:6px solid #28a745;
                padding:22px;
                border-radius:12px;
                text-align:center;
            ">
                <h2 style="color:#23864b;">😊 Positive Review</h2>
                <p style="color:#27633d;">
                    The model classified this hospital review as Positive.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="
                background:#fdeaea;
                border-left:6px solid #dc3545;
                padding:22px;
                border-radius:12px;
                text-align:center;
            ">
                <h2 style="color:#c0392b;">😟 Negative Review</h2>
                <p style="color:#8b3030;">
                    The model classified this hospital review as Negative.
                </p>
            </div>
            """, unsafe_allow_html=True)


st.divider()

st.subheader("🤖 Model Performance")

perf1, perf2, perf3 = st.columns(3)

with perf1:
    st.metric("Algorithm", "Linear SVM")

with perf2:
    st.metric("Accuracy", "86.22%")

with perf3:
    st.metric("ML Task", "Classification")

st.markdown("""
<div style="
    background:white;
    padding:18px;
    border-radius:12px;
    border:1px solid #dcebed;
">
    <h4 style="color:#087e8b;">Technologies Used</h4>
    <p style="color:#526777;">
        Python | Pandas | Scikit-learn | NLP | TF-IDF | Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("ℹ️ About MediSense AI")

st.markdown("""
<div style="
    background:#eaf4f7;
    padding:20px;
    border-radius:15px;
">
    <h4 style="color:#087e8b;">Our Project</h4>
    <p style="color:#526777;">
        MediSense AI is a healthcare review intelligence system
        that uses Natural Language Processing and Machine Learning
        to classify patient feedback into Positive and Negative sentiments.
    </p>
    <p style="color:#526777;">
        Our goal is to help understand patient experiences
        through automated review analysis.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "MediSense AI | Machine Learning & NLP Based Healthcare Review Analysis"
)




