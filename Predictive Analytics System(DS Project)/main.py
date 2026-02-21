import pandas as pd

data = {
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "orders": [120, 150, 130],
    "workers": [10, 12, 11]
}

df = pd.DataFrame(data)

print(df)
