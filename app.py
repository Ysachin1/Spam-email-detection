import streamlit as st
import joblib
import os

# Load the trained model and vectorizer
model_path = 'model/spam_classifier.pkl'
vectorizer_path = 'model/vectorizer.pkl'

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Spam Email Detector")
st.title("📧 Spam Email Detector")

st.write("Paste your email message below and click **Check** to find out if it's spam.")

user_input = st.text_area("Enter email or message here:", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.error("🚨 This message is likely **SPAM**.")
        else:
            st.success("✅ This message is **NOT spam** (HAM).")
