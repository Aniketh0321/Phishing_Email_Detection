import joblib

# Load model and vectorizer
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# User input
email = input("Enter Email Content:\n")

# Convert text into vector
email_vector = vectorizer.transform([email])

# Predict
prediction = model.predict(email_vector)

print("Prediction:", prediction[0])

# Check result
if prediction[0] == "phishing":
    print("\n⚠️ PHISHING EMAIL DETECTED")
else:
    print("\n✅ SAFE EMAIL")