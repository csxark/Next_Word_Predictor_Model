# 🧠 Next Word Predictor using LSTM

A deep learning-powered Next Word Prediction application built with **TensorFlow**, **LSTM (Long Short-Term Memory Networks)**, and **Streamlit**. The model predicts the most likely next word for a given text sequence, demonstrating the use of sequence modeling and natural language processing techniques.

<!-- ## 🚀 Live Demo

Add your deployment link here:

```text
https://your-app-url.streamlit.app
```

--- -->

## 📖 Project Overview

Next Word Prediction is a Natural Language Processing (NLP) task where a model predicts the most probable next word based on the preceding sequence of words.

This project uses an LSTM neural network trained on text data to learn contextual word relationships and generate accurate next-word predictions.

### Features

* Predicts the next word from user-provided text
* Interactive Streamlit web interface
* TensorFlow/Keras LSTM model
* Tokenizer and sequence preprocessing pipeline
* Clean and responsive UI
* Ready for deployment on Streamlit Cloud, Render, or Hugging Face Spaces

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* Pickle

---

## 📂 Project Structure

```text
Next_Word_Predictor_Model/
│
├── app.py                # Streamlit application
├── lstm_model.h5         # Trained LSTM model
├── tokenizer.pkl         # Saved tokenizer
├── max_len.pkl           # Maximum sequence length
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
├── LICENSE               # License file
└── .gitignore
```

---

## 🧠 Model Training

The complete model training notebook and experimentation process can be found on Kaggle:

### Kaggle Notebook

https://www.kaggle.com/code/csxark/next-word-predictions

The notebook includes:

* Data preprocessing
* Tokenization
* Sequence generation
* LSTM model architecture
* Training and validation
* Model saving
* Prediction pipeline

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Next_Word_Predictor_Model.git

cd Next_Word_Predictor_Model
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📦 Requirements

Example dependencies:

```txt
streamlit
tensorflow
numpy
```

Install manually:

```bash
pip install streamlit tensorflow numpy
```

---

## 🎯 Example Usage

Input:

```text
The future of artificial
```

Output:

```text
intelligence
```

---

## 🔍 How It Works

1. User enters a text sequence.
2. The text is converted into token sequences using the saved tokenizer.
3. Sequences are padded to the required length.
4. The trained LSTM model predicts probabilities for the next word.
5. The word with the highest probability is returned to the user.

---

## 📈 Future Improvements

* Top-K word predictions
* Word probability visualization
* Text generation using multiple predicted words
* Transformer-based architecture
* Support for custom datasets
* Better handling of out-of-vocabulary words

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is distributed under the terms of the LICENSE file included in this repository.

---

## 👨‍💻 Author

Ark

If you found this project useful, consider giving the repository a ⭐ on GitHub.
