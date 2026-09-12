# Formula-1-Lap-Time-
I made this for GSC by using simple linear regression model.
# F1 Lap Time Predictor 

> A machine learning project that predicts Formula 1 lap times during the **2020 Spanish Grand Prix** by modeling the impact of tire degradation over time.

---

## Project Overview
This project uses historic Ergast F1 dataset CSV files to build and evaluate predictive models for race strategy. It focuses on the **top 8 finishing drivers** of the 2020 Spanish Grand Prix, evaluating how accounting for tire life affects the accuracy of lap time predictions compared to a simple linear baseline.

# Features
* **Data Cleaning:** Automatically filters out anomalous laps and opening-lap irregularities.
* **Feature Engineering:** Calculates tire age and non-linear degradation parameters (`tire_age^2`) to simulate performance drop-offs.
* **Model Benchmarking:** Directly compares two regression approaches:
  1. **Baseline Model:** Evaluates pace strictly against the chronological lap counter.
  2. **Tire Model:** Integrates tire age metrics for highly nuanced degradation tracking.
* **Performance Evaluation:** Evaluates testing splits using root-mean-squared error (RMSE) and mean absolute error (MAE).
* **Visual Diagnostics:** Generates a real-time tracking plot comparing actual telemetry times against both predictors.

## Tech Stack & Requirements
* **Python 3**
* **Pandas** (Data manipulation)
* **Scikit-Learn** (Linear regression modeling & metrics)
* **Matplotlib** (Data visualization)

---

##  Dataset Requirements
The script expects the following CSV files (derived from the Ergast motor racing developer database) to be placed in the root directory:
* `lap_times.csv`
* `pit_stops.csv`
* `results.csv`
* `races.csv`

---

##  Installation & Usage

1. **Clone the repository and navigate inside:**
   ```bash
   git clone https://github.com
   cd f1-lap-time-predictor
   ```

2. **Install required dependencies:**
   ```bash
   pip install pandas scikit-learn matplotlib
   ```

3. **Run the predictor:**
   ```bash
   python predictor.py
   ```

---

## Example Output
When executed, the project outputs clean metrics to the terminal alongside a trend comparison graph:

```text
Removed laps: [Count]

Baseline
RMSE: [Score]
MAE: [Score]

Tire Model
RMSE: [Score]
MAE: [Score]
```

## License
Distributed under the [MIT License](LICENSE).
