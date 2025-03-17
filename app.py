import streamlit as st
import pickle

with open('model/phishing_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Phishing URL Detection")
st.write("Enter a URL below to check if it's safe or a phishing attempt.")

url_input = st.text_input("Enter URL:")

if st.button("Check URL"):
    if url_input:
        prediction = model.predict([url_input])[0]
        if prediction == 0:  # Bad URL
            st.error("⚠️ This URL is potentially a phishing site!")
        else:  # Good URL
            st.success("✅ This URL appears to be safe.")
    else:
        st.warning("Please enter a URL to check.")


st.subheader("Model Performance")
accuracy_scores = {
    "XGBoost": 0.92, 
    "Decision Tree": 0.89,
    "Random Forest": 0.91,
    "MLP": 0.88,
    "SVC": 0.86
}

sorted_scores = sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
for model_name, score in sorted_scores:
    st.write(f"*{model_name}:* {score * 100:.2f}%")
