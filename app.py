import streamlit as st
import joblib
from scipy.sparse import hstack

# Load the trained model and vectorizers
model = joblib.load("sms_spam_model.pkl")
word_tfidf = joblib.load("word_tfidf.pkl")
char_tfidf = joblib.load("char_tfidf.pkl")

st.title("Spam SMS Detector📩")
st.write("Check whether the Message is Spam or Not Spam.")

message = st.text_area("Enter your Message:")

if st.button("Check Message"):
    if message.strip():
        word_features = word_tfidf.transform([message])
        char_features = char_tfidf.transform([message])
        combined_features = hstack([word_features, char_features])

        prediction = model.predict(combined_features)[0]

        if prediction == 1:
            st.error("Spam ⚠︎")
        else:
            st.success("Not Spam ✔")
    else:
        st.warning("Please enter a message.")