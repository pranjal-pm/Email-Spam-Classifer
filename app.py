import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from scipy.sparse import csr_matrix
import time


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Set page configuration
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-container {
        background-color: #4e54c8;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title-text {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .subtitle-text {
        color: #e0e0e0;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .input-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .spam-result {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: 700;
    }
    .not-spam-result {
        color: #2ecc71;
        font-size: 2rem;
        font-weight: 700;
    }
    .stButton button {
        background-color: #4e54c8;
        color: Whilte;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #3a3f9e;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stTextArea textarea {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #777;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Title section
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.markdown('<h1 class="title-text">Email/SMS Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Detect unwanted messages with AI-powered classification</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<h2 style="color: #333; margin-bottom: 1rem;">Enter Your Message</h2>', unsafe_allow_html=True)
input_sms = st.text_area("", placeholder="Type or paste your email/SMS content here...", height=200)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('Analyze Message', key="predict_button")

# Result section
if predict_button and input_sms:
    with st.spinner('Analyzing message...'):
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        
        # Add a small delay for better UX
        time.sleep(0.5)
        
        # 4. Display result
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        if result == 1:
            st.markdown('<h2 class="spam-result">⚠️ SPAM DETECTED</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color: #e74c3c; font-size: 1.2rem;">This message appears to be spam. Be cautious!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 class="not-spam-result">✅ NOT SPAM</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color: #2ecc71; font-size: 1.2rem;">This message appears to be legitimate.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
elif predict_button and not input_sms:
    st.warning("Please enter a message to analyze.")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown('© 2023 Email/SMS Spam Classifier | Powered by Machine Learning', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)