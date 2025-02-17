import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("london_weather.csv")


def plot_average_temperature_years(df):
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    avg_temp_per_year = df.groupby('year')['mean_temp'].mean()

    plt.figure(figsize=(10, 6))
    avg_temp_per_year.plot(kind='bar', color='skyblue')

    plt.title('Average Temperature per Year in London')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')

    plt.show()


if __name__ == "__main__":
    plot_average_temperature_years(df)
