import pandas as pd
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Load the phishing data
phish_data = pd.read_csv('phishing_site_urls.csv')

# Data Preprocessing
phish_data['Label'] = phish_data['Label'].map({'bad': 0, 'good': 1})

# Split into features (X) and target (y)
X = phish_data['URL']
y = phish_data['Label']

# Split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42)

# Label Encoding for target variable (in case it's in text form)
label_encoder = LabelEncoder()
trainY_encoded = label_encoder.fit_transform(trainY)
testY_encoded = label_encoder.transform(testY)

# Create a pipeline with CountVectorizer and XGBClassifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),  # Convert text to features
    ('classifier', XGBClassifier())  # Train using XGBoost
])

# Train the model
pipeline.fit(trainX, trainY_encoded)

# Save the trained model to disk
with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Evaluate the model
train_accuracy = pipeline.score(trainX, trainY_encoded)
test_accuracy = pipeline.score(testX, testY_encoded)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Generate classification report and confusion matrix
y_pred = pipeline.predict(testX)
print('\nCLASSIFICATION REPORT\n')
print(classification_report(y_pred, testY_encoded, target_names=['Bad', 'Good']))

con_mat = confusion_matrix(y_pred, testY_encoded)
print('\nCONFUSION MATRIX')
print(con_mat)

# Save the classification report and confusion matrix
np.savetxt('confusion_matrix.csv', con_mat, delimiter=',')
