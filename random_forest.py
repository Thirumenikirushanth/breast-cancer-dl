# models/ml_random_forest.py
from preprocess import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
X_train, X_test, y_train, y_test = load_data()

# Create a pipeline to handle missing values and scaling
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # ensures no NaNs
    ('scaler', StandardScaler()),                   # optional scaling
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


