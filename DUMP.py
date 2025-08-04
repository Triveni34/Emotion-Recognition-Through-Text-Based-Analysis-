import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the data from CSV file
df = pd.read_csv('Emotiondetection.csv')

# Encode the labels
df['Emotion'] = df['Emotion'].astype('category')
label_mapping = dict(enumerate(df['Emotion'].cat.categories))
df['Emotion'] = df['Emotion'].cat.codes

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['Emotion'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train_vect, y_train)
y_pred = clf.predict(X_test_vect)

# Print the results
print(f"Results for Decision Tree:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, labels=list(label_mapping.keys()), target_names=list(label_mapping.values())))

# Save the model and vectorizer as .pkl files
joblib.dump(clf, 'decision_tree_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_mapping, 'label_mapping.pkl')
print("Model, vectorizer, and label mapping saved.")
