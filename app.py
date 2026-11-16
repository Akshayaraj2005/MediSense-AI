
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
    background-color: #f0f7f8;
}

.main-title {
    color: #087e8b;
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    color: #526777;
    text-align: center;
    font-size: 18px;
}

.review-box {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #d8e6e8;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">🏥 MediSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Healthcare Review Intelligence System</div>',
    unsafe_allow_html=True
)

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
st.subheader("Patient Review Analysis")

st.write(
    "Enter a hospital patient review below to identify its sentiment."
)

review = st.text_area(
    "Enter Patient Review",
    placeholder="Example: The doctors were helpful and the treatment was excellent...",
    height=150
)

if st.button("🔍 Analyze Review", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a hospital review.")

    else:
        cleaned_review = preprocess_review(review)
        review_tfidf = tfidf.transform([cleaned_review])
        prediction = model.predict(review_tfidf)

        st.divider()
        st.subheader("Analysis Result")

        if prediction[0] == 1:
            st.success("😊 Positive Review")
            st.write(
                "The model classified this hospital review as Positive."
            )

        else:
            st.error("😟 Negative Review")
            st.write(
                "The model classified this hospital review as Negative."
            )

st.divider()

st.caption(
    "MediSense AI | Machine Learning & NLP Based Healthcare Review Analysis"
)
