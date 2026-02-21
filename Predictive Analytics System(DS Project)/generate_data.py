import pandas as pd
import numpy as np

# date range craeted (2 years of daily data)
dates = pd.date_range(start="2023-01-01", end="2024-12-31")

# For Warehouses and regions
warehouses = ["WH_1", "WH_2", "WH_3"]
regions = ["North", "South", "East", "West"]

data = []

# To Generate data
for date in dates:
    for wh in warehouses:
        region = np.random.choice(regions)

        base_orders = np.random.randint(80, 150)

        # Increase in demand due to festive season (Due to higher demand in Oct–Dec)
        if date.month in [10, 11, 12]:
            base_orders += np.random.randint(30, 60)

        workers = int(base_orders / 12)
        shipment_weight = base_orders * np.random.uniform(1.5, 3.0)

        scanner_used = np.random.choice([0, 1], p=[0.3, 0.7])

        processing_time = np.random.uniform(25, 40)
        if scanner_used == 1:
            processing_time -= np.random.uniform(5, 10)

        data.append([
            date, wh, region, base_orders,
            shipment_weight, workers,
            scanner_used, processing_time
        ])

# Create DataFrame
columns = [
    "date", "warehouse_id", "region", "orders",
    "shipment_weight", "workers",
    "scanner_used", "processing_time"
]

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("logistics_data.csv", index=False)

print("Synthetic logistics data generated successfully!")
