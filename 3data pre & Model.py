import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"D:\House Price ML\california_housing.csv")
print(df)


X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


print(X.head())
print(y.head())

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled[:5])

print(X_train_scaled.shape)
print(X_test_scaled.shape)

# Step- 9 Train the first model (Linear Regression)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"D:\House Price ML\california_housing.csv")
print(df)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(y_pred[:10])

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(comparison.head(10))


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error :", mae)
print("Mean Squared Error  :", mse)
print("Root Mean Squared Error :", rmse)
print("R² Score :", r2)


print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# improving the modelby using Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib


dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print(dt_pred[:10])

print("Decision Tree R² Score:", r2_score(y_test, dt_pred))

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Create and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Show first 10 predictions
print("First 10 Predictions:")
print(rf_pred[:10])

# Calculate and print R² score
print("Random Forest R² Score:", r2_score(y_test, rf_pred))


print("Linear Regression :", r2_score(y_test, y_pred))
print("Decision Tree     :", r2_score(y_test, dt_pred))
print("Random Forest     :", r2_score(y_test, rf_pred))

joblib.dump(rf_model, "house_price_model.pkl")


loaded_model = joblib.load("house_price_model.pkl")

print(loaded_model)


sample_prediction = loaded_model.predict(X_test[:5])

print(sample_prediction)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("Best Cross Validation Score:")
print(grid_search.best_score_)

best_model = grid_search.best_estimator_

best_pred = best_model.predict(X_test)

print("Final R² Score:")
print(r2_score(y_test, best_pred))

print("\nModel Comparison")
print("----------------------------")
print("Linear Regression :", r2_score(y_test, y_pred))
print("Decision Tree     :", r2_score(y_test, dt_pred))
print("Random Forest     :", r2_score(y_test, rf_pred))
print("Tuned Random Forest:", r2_score(y_test, best_pred))


# Extract feature importance
feature_importance = best_model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

# Sort features
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Print feature importance
print(importance_df)

# Plot
plt.figure(figsize=(10,6))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance Score")

plt.xticks(rotation=45)

plt.show()

# Most important feature
print("Most Important Feature:")
print(importance_df.iloc[0])

# Top 5 features
print("\nTop 5 Features")
print(importance_df.head())

# Save results
importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

import joblib


# Save model
joblib.dump(best_model, "best_house_price_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Load model
loaded_model = joblib.load("best_house_price_model.pkl")

# Load scaler
loaded_scaler = joblib.load("scaler.pkl")

# New house
new_house = pd.DataFrame([[
    8.32,
    41,
    6.98,
    1.02,
    322,
    2.56,
    37.88,
    -122.23
]], columns=X.columns)

# Scale input
new_house_scaled = loaded_scaler.transform(new_house)

# Predict
new_house_df = pd.DataFrame(
    new_house_scaled,
    columns=X.columns
)

prediction = loaded_model.predict(new_house_df)

print("Predicted House Value:", prediction[0])
print(f"Predicted Price: ${prediction[0] * 100000:,.2f}")

