from sklearn.datasets import fetch_california_housing
import pandas as pd

# Download dataset
housing = fetch_california_housing(as_frame=True)

# Convert to DataFrame
df = housing.frame

# Display first 5 rows
print(df.head())

# Save as CSV
df.to_csv("california_housing.csv", index=False)

print("Dataset downloaded successfully!")


 