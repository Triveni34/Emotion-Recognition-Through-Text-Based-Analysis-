import joblib

# Load the saved model, vectorizer, and label mapping
clf = joblib.load('decision_tree_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_mapping = joblib.load('label_mapping.pkl')

# Example usage
new_text = input("Enter a text")
new_texts=[new_text]
new_texts_vect = vectorizer.transform(new_texts)
predictions = clf.predict(new_texts_vect)
print(f"Predictions: {predictions[0]}")

# Map predictions to emotion names using label_mapping
predicted_emotions = [label_mapping[pred] for pred in predictions]
print(predicted_emotions[0])
