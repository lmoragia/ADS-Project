#Import Libraries
import pandas as pd
import numpy as np
import re

# Import dataset from CSV
data = pd.read_csv('Iris.csv')
data.head()

data.info()

# Rename the values in the 'species' column
data['Species'] = data['Species'].replace({
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica': 3
})

data.head()

# Save the cleaned DataFrame to a new CSV file
data.to_csv('cleaned_Iris_data.csv', index=False)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.drop('Species', axis=1)), columns=data.columns[:-1])
data_scaled.head(5)

# Save the scaled DataFrame to a new CSV file
data_scaled.to_csv('data_scaled.csv', index=False)

#Split Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# Separate features and target
X = data_scaled
y = data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("Training Set Shapes:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("\nTesting Set Shapes:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#Train and Evaluate Models
#Import classifiers and train models:
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machines": SVC(),
    "Decision Trees": DecisionTreeClassifier(),
    "Random Forests": RandomForestClassifier()
}

# Reshape X_train to a 2D array
#X_train = X_train.reshape(1, -1)

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test)}")

%%time
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Best model: {best_model}")
print(f"Best score: {grid.best_score_}")

%%time
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score

# Instantiate the best model from Step 4 (e.g., Random Forests)
best_model = RandomForestClassifier(n_estimators=200)

# Create the RFECV object and fit it to the training data
selector = RFECV(best_model, step=1, cv=5, scoring='accuracy')
selector.fit(X_train, y_train)

# Get the selected features and their ranks
selected_features = X_train.columns[selector.support_]
feature_ranks = selector.ranking_

print(f"Selected features: {selected_features}")
print(f"Feature ranks: {feature_ranks}")

# Convert selected_features to a list
selected_features_list = selected_features.tolist()

# Remove target variable from the list of selected features if it's present
#None

# Create new dataframes with only the selected features
X_train_selected = X_train[selected_features_list]
X_test_selected = X_test[selected_features_list]

# Save X_train_selected as a CSV file
pd.DataFrame(X_train_selected).to_csv('X_train_selected.csv', index=False)

best_model=best_model.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test_selected)

# Evaluate the model using accuracy_score
from sklearn.metrics import accuracy_score

test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy with selected features: {test_accuracy}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Make predictions on the test set
y_pred = best_model.predict(X_test_selected)

# Calculate evaluation scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Use macro, micro, or weighted
recall = recall_score(y_test, y_pred, average='macro')        # Use macro, micro, or weighted
f1 = f1_score(y_test, y_pred, average='macro')                # Use macro, micro, or weighted

# For roc_auc_score, if it's multiclass, you need to use the `multi_class` parameter
if len(set(y_test)) > 2:
    auc_roc = roc_auc_score(y_test, best_model.predict_proba(X_test_selected), multi_class='ovr')  # or 'ovo'
else:
    auc_roc = roc_auc_score(y_test, y_pred)

# Print evaluation scores
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


# Save the confusion matrix as an image

plt.savefig('confusion_matrix.png')

import json
selected_features_list = selected_features.tolist()

with open("selected_features.json", "w") as f:
    json.dump(selected_features_list, f)


import joblib

# Save the best model to a file
joblib.dump(best_model, "best_model1.pkl")


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load the trained model and scaler
model = joblib.load("best_model1.pkl")


# Define the app title and layout
st.title("Iris Flower Species App")

# Define input fields for features
id = st.number_input("Id", min_value=0, max_value=100, value=60, step=1)
sepal_length_cm = st.number_input("Sepal Length in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width_cm = st.number_input("Sepal Width in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
petal_length_cm = st.number_input("Petal Length in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
petal_width_cm = st.number_input("Petal Width in CM", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "Id": [id],
            "SepalLengthCm": [sepal_length_cm],
            "SepalWidthCm": [sepal_width_cm],
            "PetalLengthCm": [petal_length_cm],
            "PetalWidthCm": [petal_width_cm]
        }
    )

    # Scale input data using the scaler used during training
    input_data_scaled = scaler.transform(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Unhealthy Flower.")
    else:
        st.success("Healthy Flower.")
