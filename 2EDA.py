import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r"D:\House Price ML\california_housing.csv")
print(df)

print(df.shape)

df.hist(figsize=(15,10), bins=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))

sns.histplot(df["MedHouseVal"], bins=40, kde=True)

plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Frequency")

plt.show()

# Correlation Matrix
correlation = df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,6))


sns.scatterplot(
    x="MedInc",
    y="MedHouseVal",
    data=df
)

plt.show()

plt.figure(figsize=(10,5))

sns.boxplot(x=df["MedHouseVal"])

plt.show()

sns.pairplot(
    df[
        [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "MedHouseVal"
        ]
    ]
)

plt.show()


plt.figure(figsize=(10,8))

plt.scatter(
    df["Longitude"],
    df["Latitude"],
    alpha=0.3,
    c=df["MedHouseVal"],
    cmap="viridis"
)

plt.colorbar(label="House Price")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("California Housing Map")

plt.show()

print(df.describe())