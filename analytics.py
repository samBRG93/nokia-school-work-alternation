import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

df = pd.read_csv("london_weather.csv")

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Calculate the average temperature per year
avg_temp_per_year = df.groupby('year')['mean_temp'].mean()

# Plot the bar chart
plt.figure(figsize=(10,6))
avg_temp_per_year.plot(kind='bar', color='skyblue')

# Adding title and labels
plt.title('Average Temperature per Year in London')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')

# Display the plot
plt.show()

# plt.figure(figsize=(8, 5))
#     plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
#     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="dashed")  # Linea ideale
#     plt.xlabel("Expected Value")
#     plt.ylabel("Prediction")
#     plt.title("Predictions vs Expected Values")
#     plt.grid(True)
#     plt.savefig('predictions.png')