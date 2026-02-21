import pandas as pd

# Load dataset
df = pd.read_csv("logistics_data.csv")

# Convert date to datetime
df["date"] = pd.to_datetime(df["date"])

print(df.head())

# Extract useful date features
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Drop original date column
df = df.drop(columns=["date"])
# Convert categorical columns to numeric
df = pd.get_dummies(df, columns=["warehouse_id", "region"], drop_first=True)

print(df.head())

# Target variable
y = df["orders"]

# Features (everything except orders)
X = df.drop(columns=["orders"])

# Removing workers to avoid feature leakage
X = X.drop(columns=["workers"])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Predict on test data
y_pred = model.predict(X_test)

print("Sample predictions:", y_pred[:5])

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

#now,i'm using random forest method 
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
print("Random Forest training completed!")
rf_pred = rf_model.predict(X_test)
print("Random Forest sample predictions:", rf_pred[:5])
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
#compare rf & linear regression
print("\nMODEL COMPARISON")
print("----------------")
print("Linear Regression  -> MAE:", mae, " RMSE:", rmse)
print("Random Forest      -> MAE:", rf_mae, " RMSE:", rf_rmse)

# Feature importance from Random Forest
import pandas as pd

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

# Sort by importance (i've given importance to months & shipment weight as everything depends on orders)
feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
)

print("\nFEATURE IMPORTANCE")
print("------------------")
print(feature_importance)

import matplotlib.pyplot as plt

feature_importance.head(8).plot(
    x="Feature",
    y="Importance",
    kind="bar",
    legend=False
)

#visual expl.
plt.title("Top Features Influencing Order Demand")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
