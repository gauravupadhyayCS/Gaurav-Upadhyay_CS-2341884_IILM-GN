from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# =========================
# Load data and prepare model (runs once at startup)
# =========================
DATA_PATH = "Automobile_dataset.csv"

df = pd.read_csv(DATA_PATH)

selected_features = ['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'MSRP']

missing_cols = [c for c in selected_features if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in CSV: {missing_cols}")

df = df.dropna(subset=selected_features)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[selected_features])

n_neighbors = min(50, len(df))
if n_neighbors < 1:
    raise ValueError("Not enough data")

model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
model.fit(data_scaled)

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None

    # Default values for initial form
    default_values = {
        "hp": 200,
        "highway_mpg": 30,
        "cylinders": 4,
        "city_mpg": 25,
        "budget": 30000
    }

    if request.method == "POST":
        # Get form values
        hp = float(request.form.get("hp", default_values["hp"]))
        highway_mpg = float(request.form.get("highway_mpg", default_values["highway_mpg"]))
        cylinders = float(request.form.get("cylinders", default_values["cylinders"]))
        city_mpg = float(request.form.get("city_mpg", default_values["city_mpg"]))
        budget = float(request.form.get("budget", default_values["budget"]))

        # Prepare user input
        user_input = np.array([[hp, cylinders, highway_mpg, city_mpg, budget]])
        user_scaled = scaler.transform(user_input)

        distances, indices = model.kneighbors(user_scaled)

        candidates = df.iloc[indices[0]].copy()
        candidates["Distance"] = distances[0]

        max_dist = candidates["Distance"].max()
        if max_dist > 0:
            candidates["Similarity Score"] = 1 - (candidates["Distance"] / max_dist)
        else:
            candidates["Similarity Score"] = 1.0

        # Filter by budget
        filtered = candidates[candidates["MSRP"] <= budget]

        if not filtered.empty:
            filtered = (
                filtered.sort_values("Similarity Score", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

            # Only keep needed columns for template
            recommendations = filtered[
                ["Make", "Model", "Year", "Engine Fuel Type", "Engine HP", "MSRP", "Similarity Score"]
            ].round({"Similarity Score": 3})

    return render_template("index.html", recommendations=recommendations, defaults=default_values)


if __name__ == "__main__":
    app.run(debug=True)
