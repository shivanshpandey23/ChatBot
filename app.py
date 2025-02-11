import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("chatbot_model.pkl")  # Ensure this file exists
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Function to get chatbot response
def chatbot_response(user_input):
    user_input_cleaned = user_input.lower()
    input_vector = vectorizer.transform([user_input_cleaned])
    predicted_label = model.predict(input_vector)[0]
    intent = label_encoder.inverse_transform([predicted_label])[0]
    return intent

# Streamlit UI
st.title("Customer Support Chatbot 🤖")
st.write("Type your question below and get an instant response!")

# User input box
user_input = st.text_input("Enter your message:")

if st.button("Submit"):
    if user_input:
        bot_response = chatbot_response(user_input)
        st.subheader("Chatbot Response:")
        st.write(bot_response)
