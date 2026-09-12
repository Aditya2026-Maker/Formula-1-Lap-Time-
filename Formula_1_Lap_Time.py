import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Files load
laps = pd.read_csv("lap_times.csv")
pits = pd.read_csv("pit_stops.csv")
results = pd.read_csv("results.csv")
races = pd.read_csv("races.csv")

# 2020 Spanish GP
race_id = races[
    (races.year == 2020) &
    (races.name == "Spanish Grand Prix")
].raceId.iloc[0]

# Top 8 drivers
drivers = results[results.raceId == race_id] \
    .sort_values("positionOrder").head(8).driverId.tolist()

# Race lap data
data = laps[
    (laps.raceId == race_id) &
    (laps.driverId.isin(drivers))
].copy()

# Convert milliseconds to seconds
data["lap_time"] = data["milliseconds"] / 1000

# cleaning
old = len(data)
data = data[(data.lap != 1) & (data.lap_time > 0)]

print("Removed laps:", old - len(data))

# Tire age
data["tire_age"] = data["lap"] - 1

# Tire age squared
data["tire_age2"] = data["tire_age"] ** 2

# Train/Test split
train = data.iloc[:int(len(data) * 0.8)]
test = data.iloc[int(len(data) * 0.8):]

y_train = train["lap_time"]
y_test = test["lap_time"]

# Model 1: Baseline
model1 = LinearRegression()
model1.fit(train[["lap"]], y_train)

pred1 = model1.predict(test[["lap"]])

# Model 2: Tire Age
model2 = LinearRegression()
model2.fit(train[["lap", "tire_age", "tire_age2"]], y_train)

pred2 = model2.predict(
    test[["lap", "tire_age", "tire_age2"]]
)

# Results
print("\nBaseline")
print("RMSE:", round(mean_squared_error(y_test, pred1) ** 0.5, 2))
print("MAE:", round(mean_absolute_error(y_test, pred1), 2))

print("\nTire Model")
print("RMSE:", round(mean_squared_error(y_test, pred2) ** 0.5, 2))
print("MAE:", round(mean_absolute_error(y_test, pred2), 2))

# Graph
plt.plot(y_test.values, label="Actual")
plt.plot(pred1, label="Baseline")
plt.plot(pred2, label="Tire Model")

plt.xlabel("Lap")
plt.ylabel("Lap Time (seconds)")
plt.title("Actual vs Predicted Lap Times")
plt.legend()
plt.show()