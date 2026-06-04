import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Background */

.stApp {
    background: linear-gradient(
        135deg,
        #023047 0%,
        #0b3d5c 100%
    );
}

/* Title */

.main-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 800;
    color: #8ECAE6;
    margin-bottom: 0.2rem;
}

/* Subtitle */

.subtitle {
    text-align: center;
    color: #d6edf6;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Input Label */

.stTextInput label {
    color: white !important;
    font-size: 1rem;
    font-weight: 500;
}

/* Input Box */

.stTextInput input {
    background-color: rgba(255,255,255,0.08);
    color: white !important;
    border: 2px solid #219EBC;
    border-radius: 12px;
    padding: 12px;
}

.stTextInput input:focus {
    border-color: #FFB703;
    box-shadow: 0 0 15px rgba(255,183,3,0.5);
}

/* Placeholder */

.stTextInput input::placeholder {
    color: #b9d8e6;
}

/* Button */

.stButton button {
    width: 100%;
    height: 55px;

    background: linear-gradient(
        90deg,
        #219EBC,
        #8ECAE6
    );

    color: #023047;
    font-size: 18px;
    font-weight: bold;

    border: none;
    border-radius: 12px;

    transition: all 0.3s ease;
}

.stButton button:hover {
    background: linear-gradient(
        90deg,
        #FFB703,
        #FB8500
    );

    color: white;
    transform: translateY(-2px);
}

/* Prediction Card */

.prediction-card {
    background: rgba(255,255,255,0.08);

    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.15);

    border-radius: 20px;

    padding: 40px;

    margin-top: 30px;

    text-align: center;

    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

/* Card Heading */

.prediction-card h3 {
    color: #8ECAE6;
    font-size: 1.4rem;
    margin-bottom: 20px;
}

/* Predicted Word */

.prediction-card h1 {
    color: #FFB703;
    font-size: 4rem;
    font-weight: 800;
    margin: 0;
}

/* Info Box */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Footer */

.footer {
    text-align: center;
    color: #8ECAE6;
    margin-top: 50px;
    opacity: 0.8;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

model, tokenizer, max_len = load_resources()

# ---------------- PREDICT FUNCTION ---------------- #
def predict_next_word(text):
    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    preds = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(preds)

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return "Unknown"

# ---------------- UI ---------------- #

st.markdown("""
<div class="main-title">
🧠 Next Word Predictor
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Predict the next word using a trained LSTM neural network
</div>
""", unsafe_allow_html=True)

st.info(
    "💡 Enter a phrase or sentence and the model will predict the most likely next word."
)

user_input = st.text_input(
    "Enter your sentence",
    placeholder="Example: The future of artificial intelligence"
)

if st.button("🚀 Predict Next Word"):

    if not user_input.strip():
        st.warning("Please enter some text.")

    else:
        with st.spinner("Thinking..."):
            next_word = predict_next_word(user_input)

        st.markdown(
            f"""
            <div class="prediction-card">
                <h3>✨ Predicted Next Word</h3>
                <h1>{next_word}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

