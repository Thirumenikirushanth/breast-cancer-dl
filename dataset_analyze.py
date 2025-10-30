import pandas as pd

data = pd.read_csv(r"C:\Users\Computer Corner\Desktop\Breast Cancer DL\data.csv")  # adjust path if needed
print("Data shape:", data.shape)
print(data.head())
