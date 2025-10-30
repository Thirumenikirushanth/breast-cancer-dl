# models/ml_svm.py
from preprocess import load_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
X_train, X_test, y_train, y_test = load_data()

# Create a pipeline for imputation, scaling, and SVM
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # handle missing values
    ('scaler', StandardScaler()),                   # scale features
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
