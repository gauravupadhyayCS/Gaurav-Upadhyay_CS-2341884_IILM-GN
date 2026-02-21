import pandas as pd

# Load the CSV file
df = pd.read_csv("logistics_data.csv")
df["date"] = pd.to_datetime(df["date"])
print("Data loaded successfully!")
print(df.head())
print(df.tail())
df.info()
print(df.describe())
print("Average orders:", df["orders"].mean())
print(df.groupby("region")["orders"].mean())
#scanners reduce processing time (reduce the time by around 10 minutes in each processing)
print(df.groupby("scanner_used")["processing_time"].mean())

# Graph
import matplotlib.pyplot as plt

df.groupby("region")["orders"].mean().plot(kind="bar")
plt.title("Average Orders by Region")
plt.ylabel("Orders")
plt.xlabel("Region")

plt.savefig("average_orders_by_region.png")  # saves image
plt.close()


