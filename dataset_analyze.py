# analyze_dataset.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv(r"C:\Users\Computer Corner\Desktop\Breast Cancer DL\data.csv")

# Basic info
print("✅ Dataset loaded successfully!")
print("Shape of data:", data.shape)
print("\n🔹 First 5 rows:")
print(data.head())

# Summary statistics
print("\n🔹 Data Info:")
print(data.info())

print("\n🔹 Summary Statistics:")
print(data.describe())

# Check for missing values
print("\n🔹 Missing Values:")
print(data.isnull().sum())

# Check target variable distribution
if 'diagnosis' in data.columns:
    print("\n🔹 Target Class Distribution:")
    print(data['diagnosis'].value_counts())

    # Plot target distribution
    plt.figure(figsize=(5,4))
    sns.countplot(x='diagnosis', data=data, palette='Set2')
    plt.title("Target Variable Distribution")
    plt.show()

# Correlation heatmap (if numerical columns exist)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# Example pairplot (optional, can be slow)
# sns.pairplot(data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']], hue='diagnosis')
# plt.show()

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\n🔹 Number of duplicate rows: {duplicates}")

# Basic statistical insight
if 'diagnosis' in data.columns:
    malignant = data[data['diagnosis'] == 'M']
    benign = data[data['diagnosis'] == 'B']
    print(f"\nAverage radius (Malignant): {malignant['radius_mean'].mean():.2f}")
    print(f"Average radius (Benign): {benign['radius_mean'].mean():.2f}")

