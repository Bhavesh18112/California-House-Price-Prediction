import pandas as pd

df = pd.read_csv(r"D:\House Price ML\california_housing.csv")
print(df)
#"D:\House Price ML\california_housing.csv"


print(df.head())

print(df.tail())

print(df.shape)

print(df.info())

print(df.describe())

print(df.columns)

print(df.dtypes)


# Check missing values
print(df.isnull().sum())

# Show rows with missing values
print(df[df.isnull().any(axis=1)])

# Percentage of missing values
print((df.isnull().sum() / len(df)) * 100)


# Verify missing values
print(df.isnull().sum())

# Check duplicates
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Check data types
print(df.dtypes)

# Dataset shape
print(df.shape)