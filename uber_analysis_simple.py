# ============================================================
# Uber Driver Demand Analysis
# MDST Winter 2026
#
# HOW TO USE:
#   1. Put your CSV file in the same folder as this script
#   2. Change FILE_NAME below to match your file's name
#   3. Run the script!
#
# This will answer: "Given the time and day, how many
# drivers should Uber expect to need?"
# ============================================================

# ----- CHANGE THIS to your CSV file name -----
FILE_NAME = "UberDataset_Current__1_.csv"
# ----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Where to save our output charts and files
OUTPUT_FOLDER = "uber_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading data...")

# ============================================================
# STEP 1: Load and clean the data
# ============================================================

df = pd.read_csv(FILE_NAME, index_col=0)

# Convert date columns from text to actual dates
df["START_DATE"] = pd.to_datetime(df["START_DATE"])
df["END_DATE"]   = pd.to_datetime(df["END_DATE"])

# Calculate how long each trip took in minutes
df["DURATION_MIN"] = (df["END_DATE"] - df["START_DATE"]).dt.total_seconds() / 60

# Replace "NO_DATA" in PURPOSE with something cleaner
df["PURPOSE"] = df["PURPOSE"].replace("NO_DATA", "Unknown")

# Remove any rows where the trip duration doesn't make sense
df = df[df["DURATION_MIN"] > 0]

# Add a readable day name column (e.g. 0 -> "Mon")
day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df["day_name"] = df["day_of_week"].map(day_map)

print(f"  Loaded {len(df)} trips from {df['START_DATE'].min().date()} to {df['START_DATE'].max().date()}")

# ============================================================
# STEP 2: Count how many trips happen at each day + hour
#         This is our "demand" — how busy Uber is
# ============================================================

demand = df.groupby(["day_of_week", "hour"]).size().reset_index(name="trip_count")
demand["day_name"] = demand["day_of_week"].map(day_map)

print(f"  Built demand table with {len(demand)} day/hour combinations")

# ============================================================
# STEP 3: Exploratory Data Analysis (EDA)
#         Just looking at the data with some basic charts
# ============================================================

print("\nMaking EDA charts...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Uber Trip Data - Overview", fontsize=14, fontweight="bold")

# Chart 1: How many trips happen each hour?
ax = axes[0, 0]
hourly_counts = df.groupby("hour").size()
ax.bar(hourly_counts.index, hourly_counts.values, color="steelblue")
ax.set_title("Trips by Hour of Day")
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Trips")
# Shade the rush hour windows
ax.axvspan(6, 10, alpha=0.15, color="orange", label="AM Rush (6-10am)")
ax.axvspan(16, 20, alpha=0.15, color="red",    label="PM Rush (4-8pm)")
ax.legend(fontsize=7)

# Chart 2: How many trips happen each day of the week?
ax = axes[0, 1]
day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
daily_counts = df.groupby("day_name").size().reindex(day_order)
bar_colors = ["coral" if d in ["Sat", "Sun"] else "steelblue" for d in day_order]
ax.bar(day_order, daily_counts.values, color=bar_colors)
ax.set_title("Trips by Day of Week")
ax.set_xlabel("Day")
ax.set_ylabel("Number of Trips")
ax.text(0.98, 0.95, "Orange = Weekend", transform=ax.transAxes,
        ha="right", va="top", fontsize=7, color="coral")

# Chart 3: How far are the trips?
ax = axes[0, 2]
ax.hist(df["MILES"].clip(upper=30), bins=25, color="mediumpurple", edgecolor="white", linewidth=0.3)
ax.set_title("Trip Distance (capped at 30 miles)")
ax.set_xlabel("Miles")
ax.set_ylabel("Number of Trips")
median_miles = df["MILES"].median()
ax.axvline(median_miles, color="red", linestyle="--", label=f"Median: {median_miles:.1f} mi")
ax.legend(fontsize=8)

# Chart 4: Heatmap — which day+hour combos are busiest?
ax = axes[1, 0]
pivot = demand.pivot(index="day_of_week", columns="hour", values="trip_count").fillna(0)
pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.2, cbar_kws={"shrink": 0.8})
ax.set_title("Demand Heatmap (Day vs Hour)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Day")

# Chart 5: What are people using Uber for?
ax = axes[1, 1]
purpose_counts = df["PURPOSE"].value_counts().head(7)
ax.pie(purpose_counts.values, labels=purpose_counts.index, autopct="%1.0f%%",
       startangle=140, colors=plt.cm.Set2.colors)
ax.set_title("Trip Purpose Breakdown")

# Chart 6: Does temperature affect trip length?
ax = axes[1, 2]
# Remove outlier miles for a cleaner plot
clean = df[df["MILES"] < 25].dropna(subset=["TEMPERATURE"])
ax.scatter(clean["TEMPERATURE"], clean["MILES"], alpha=0.3, s=10, color="teal")
ax.set_title("Temperature vs Trip Distance")
ax.set_xlabel("Temperature (°F)")
ax.set_ylabel("Miles")

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/1_eda_charts.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_FOLDER}/1_eda_charts.png")

# ============================================================
# STEP 4: Statistical Analysis
#         Bootstrap confidence intervals — basically asking
#         "how sure are we about the average demand per hour?"
# ============================================================

print("\nRunning statistical analysis...")

np.random.seed(42)
N_BOOTSTRAP = 500  # number of times to resample

hourly_stats = {}
for hour, group in demand.groupby("hour"):
    values = group["trip_count"].values
    # Resample the data many times to estimate uncertainty
    bootstrap_means = []
    for _ in range(N_BOOTSTRAP):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    hourly_stats[hour] = {
        "mean":  np.mean(values),
        "lower": np.percentile(bootstrap_means, 2.5),   # 95% CI lower bound
        "upper": np.percentile(bootstrap_means, 97.5),  # 95% CI upper bound
    }

stats_df = pd.DataFrame(hourly_stats).T.reset_index().rename(columns={"index": "hour"})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Statistical Analysis", fontsize=13, fontweight="bold")

# Plot mean demand with shaded confidence interval
ax = axes[0]
ax.fill_between(stats_df["hour"], stats_df["lower"], stats_df["upper"],
                alpha=0.3, color="steelblue", label="95% Confidence Interval")
ax.plot(stats_df["hour"], stats_df["mean"], color="steelblue",
        linewidth=2, marker="o", markersize=4, label="Average demand")
ax.set_title("Average Demand per Hour (with uncertainty)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg Trips")
ax.legend()
ax.grid(True, alpha=0.3)

# Simple correlation heatmap for numeric columns
ax = axes[1]
num_cols = ["hour", "day_of_week", "MILES", "DURATION_MIN",
            "TEMPERATURE", "WEATHER_STRESS", "is_rush_hour"]
# Convert bool to int so correlation works
corr_data = df[num_cols].copy()
corr_data["is_rush_hour"] = corr_data["is_rush_hour"].astype(int)
corr = corr_data.corr()
sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, annot=True,
            fmt=".1f", annot_kws={"size": 7}, linewidths=0.5)
ax.set_title("Feature Correlations")
ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/2_statistical_analysis.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_FOLDER}/2_statistical_analysis.png")

# Print some quick stats
peak_hour = int(stats_df.loc[stats_df["mean"].idxmax(), "hour"])
quiet_hour = int(stats_df.loc[stats_df["mean"].idxmin(), "hour"])
print(f"  Busiest hour of day: {peak_hour}:00")
print(f"  Quietest hour of day: {quiet_hour}:00")
print(f"  Average trip: {df['MILES'].mean():.1f} miles, {df['DURATION_MIN'].mean():.0f} minutes")

# ============================================================
# STEP 5: Demand Forecasting
#         Smooth out the demand numbers and figure out
#         how many drivers would be needed each slot
# ============================================================

print("\nBuilding demand forecast...")

# Make a complete grid of all day+hour combos (7 days x 24 hours = 168 slots)
all_days  = list(range(7))
all_hours = list(range(24))
full_grid = pd.DataFrame(
    [(d, h) for d in all_days for h in all_hours],
    columns=["day_of_week", "hour"]
)

# Merge in the actual trip counts (some slots may have no data -> fill with 0)
full_grid = full_grid.merge(demand[["day_of_week", "hour", "trip_count"]],
                             on=["day_of_week", "hour"], how="left").fillna(0)

# Smooth the counts a little using a rolling average
full_grid["smooth_count"] = full_grid["trip_count"].rolling(window=3, min_periods=1, center=True).mean()

# Add a 15% safety buffer for the driver recommendation
SAFETY_BUFFER = 1.15
full_grid["recommended_drivers"] = (full_grid["smooth_count"] * SAFETY_BUFFER).round().astype(int)
full_grid["day_name"] = full_grid["day_of_week"].map(day_map)

fig, axes = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle("Demand Forecasting", fontsize=13, fontweight="bold")

# Line chart — one line per day
ax = axes[0]
colors_by_day = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b", "#e377c2"]
for i, day in enumerate(day_order):
    subset = full_grid[full_grid["day_name"] == day]
    linestyle = "--" if day in ["Sat", "Sun"] else "-"
    ax.plot(subset["hour"], subset["smooth_count"],
            label=day, color=colors_by_day[i], linestyle=linestyle, linewidth=1.8)
ax.set_title("Smoothed Trip Demand by Hour (each day of week)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Trip Count")
ax.legend(ncol=7, fontsize=8, loc="upper left")
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)

# Heatmap of recommended drivers
ax = axes[1]
driver_pivot = full_grid.pivot(index="day_name", columns="hour",
                                values="recommended_drivers").reindex(day_order)
im = ax.imshow(driver_pivot.values, cmap="YlGn", aspect="auto")
plt.colorbar(im, ax=ax, label="Recommended Drivers")
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), fontsize=7)
ax.set_yticks(range(7))
ax.set_yticklabels(day_order)
ax.set_title("Recommended Drivers per Slot (15% safety buffer added)")
ax.set_xlabel("Hour of Day")

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/3_demand_forecast.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_FOLDER}/3_demand_forecast.png")

# ============================================================
# STEP 6: Machine Learning
#         Train a Random Forest model to predict how many
#         trips will happen given the day and hour
# ============================================================

print("\nTraining machine learning model...")

# Build the ML dataset — one row per (day, hour) slot
weather_avgs = (df.groupby(["day_of_week", "hour"])
                  .agg(
                      avg_temp    = ("TEMPERATURE",    "mean"),
                      avg_weather = ("WEATHER_STRESS", "mean"),
                      avg_wind    = ("WIND_SPEED_10M", "mean"),
                      avg_rain    = ("RAIN",           "mean"),
                      avg_miles   = ("MILES",          "mean")
                  ).reset_index())

ml_data = demand.merge(weather_avgs, on=["day_of_week", "hour"], how="left").fillna(0)

# Add a few helpful features
ml_data["is_weekend"] = (ml_data["day_of_week"] >= 5).astype(int)
ml_data["is_rush"]    = ml_data["hour"].apply(lambda h: int(6 <= h < 10 or 16 <= h < 20))

# Use sine/cosine to capture the cyclical nature of time
# (e.g. hour 23 and hour 0 are actually close together)
ml_data["sin_hour"] = np.sin(2 * np.pi * ml_data["hour"] / 24)
ml_data["cos_hour"] = np.cos(2 * np.pi * ml_data["hour"] / 24)
ml_data["sin_day"]  = np.sin(2 * np.pi * ml_data["day_of_week"] / 7)
ml_data["cos_day"]  = np.cos(2 * np.pi * ml_data["day_of_week"] / 7)

FEATURES = ["sin_hour", "cos_hour", "sin_day", "cos_day",
            "is_weekend", "is_rush",
            "avg_temp", "avg_weather", "avg_wind", "avg_rain", "avg_miles"]

X = ml_data[FEATURES].values
y = ml_data["trip_count"].values

# Train the Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
model.fit(X, y)

# Cross-validation — test how well the model generalizes
cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
cv_r2  =  cross_val_score(model, X, y, cv=5, scoring="r2")

print(f"  Model accuracy (5-fold cross validation):")
print(f"    Mean Absolute Error: {cv_mae.mean():.2f} trips  (+/- {cv_mae.std():.2f})")
print(f"    R² Score:            {cv_r2.mean():.3f}        (+/- {cv_r2.std():.3f})")

# Use the individual trees in the forest to get a confidence range
all_tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
pred_mean = all_tree_predictions.mean(axis=0)

# Feature importances — which inputs matter most?
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": importances
}).sort_values("importance", ascending=True)

# ============================================================
# STEP 7: Predict for every day+hour slot in the week
# ============================================================

print("\nGenerating predictions for full week...")

# Build a prediction table for all 168 slots (7 days x 24 hours)
prediction_rows = []
for dow in range(7):
    for hr in range(24):
        prediction_rows.append({
            "day_of_week": dow,
            "hour":        hr,
            "is_weekend":  int(dow >= 5),
            "is_rush":     int(6 <= hr < 10 or 16 <= hr < 20),
            "avg_temp":    df["TEMPERATURE"].mean(),
            "avg_weather": df["WEATHER_STRESS"].mean(),
            "avg_wind":    df["WIND_SPEED_10M"].mean(),
            "avg_rain":    df["RAIN"].mean(),
            "avg_miles":   df["MILES"].mean(),
            "sin_hour":    np.sin(2 * np.pi * hr / 24),
            "cos_hour":    np.cos(2 * np.pi * hr / 24),
            "sin_day":     np.sin(2 * np.pi * dow / 7),
            "cos_day":     np.cos(2 * np.pi * dow / 7),
        })

predictions = pd.DataFrame(prediction_rows)
X_pred = predictions[FEATURES].values

# Get predictions + confidence intervals from each tree
tree_preds = np.array([tree.predict(X_pred) for tree in model.estimators_])
predictions["predicted_trips"]      = tree_preds.mean(axis=0).clip(min=0)
predictions["confidence_low"]       = np.percentile(tree_preds, 5,  axis=0).clip(min=0)
predictions["confidence_high"]      = np.percentile(tree_preds, 95, axis=0).clip(min=0)
predictions["recommended_drivers"]  = (predictions["predicted_trips"] * SAFETY_BUFFER).round().astype(int)
predictions["day_name"]             = predictions["day_of_week"].map(day_map)

# ============================================================
# STEP 8: Plot ML Results
# ============================================================

print("\nPlotting ML results...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Machine Learning Results", fontsize=13, fontweight="bold")

# Chart 1: Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y, pred_mean, alpha=0.6, s=25, color="steelblue")
max_val = max(y.max(), pred_mean.max()) * 1.05
ax.plot([0, max_val], [0, max_val], color="red", linestyle="--", linewidth=1.5, label="Perfect prediction")
ax.set_title("Actual vs Predicted Trips")
ax.set_xlabel("Actual Trips")
ax.set_ylabel("Predicted Trips")
ax.legend(fontsize=8)
mae = mean_absolute_error(y, pred_mean)
r2  = r2_score(y, pred_mean)
ax.text(0.05, 0.90, f"MAE = {mae:.2f}  |  R² = {r2:.2f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor="lightyellow", edgecolor="gray", boxstyle="round"))

# Chart 2: Feature importances
ax = axes[0, 1]
ax.barh(feat_imp_df["feature"], feat_imp_df["importance"], color="mediumseagreen")
ax.set_title("What Factors Matter Most? (Feature Importance)")
ax.set_xlabel("Importance Score")
ax.axvline(0, color="gray", linewidth=0.8)

# Chart 3: Friday forecast with confidence interval
ax = axes[1, 0]
friday = predictions[predictions["day_of_week"] == 4]  # 4 = Friday
ax.fill_between(friday["hour"], friday["confidence_low"], friday["confidence_high"],
                alpha=0.25, color="steelblue", label="90% Confidence Range")
ax.plot(friday["hour"], friday["predicted_trips"],
        color="steelblue", linewidth=2, marker="o", markersize=4, label="Predicted trips")
ax.plot(friday["hour"], friday["recommended_drivers"],
        color="darkorange", linewidth=1.8, linestyle="--", label="Drivers needed (+15%)")
ax.set_title("Friday: Predicted Demand + How Many Drivers to Have Ready")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Count")
ax.legend(fontsize=8)
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)

# Chart 4: Full-week driver recommendation heatmap
ax = axes[1, 1]
driver_heatmap = predictions.pivot(index="day_name", columns="hour",
                                    values="recommended_drivers").reindex(day_order)
im = ax.imshow(driver_heatmap.values, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Drivers Needed")
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), fontsize=6)
ax.set_yticks(range(7))
ax.set_yticklabels(day_order)
ax.set_title("Full Week: How Many Drivers Should Be Ready?")
ax.set_xlabel("Hour of Day")

plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/4_ml_results.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUT_FOLDER}/4_ml_results.png")

# ============================================================
# STEP 9: Save results and print summary
# ============================================================

# Save the recommendations to a CSV file
output_csv = predictions[["day_name", "hour", "predicted_trips",
                            "confidence_low", "confidence_high", "recommended_drivers"]]
output_csv.columns = ["Day", "Hour", "Predicted Trips",
                       "Low Estimate (90% CI)", "High Estimate (90% CI)", "Recommended Drivers"]
output_csv = output_csv.round(1)
output_csv.to_csv(f"{OUTPUT_FOLDER}/driver_recommendations.csv", index=False)
print(f"  Saved: {OUTPUT_FOLDER}/driver_recommendations.csv")

# Find the busiest and quietest slots
peak  = predictions.loc[predictions["predicted_trips"].idxmax()]
quiet = predictions.loc[predictions["predicted_trips"].idxmin()]

print("\n" + "="*55)
print("  RESULTS SUMMARY")
print("="*55)
print(f"  Dataset: {len(df)} trips  ({df['START_DATE'].min().date()} to {df['START_DATE'].max().date()})")
print()
print(f"  Busiest slot:  {peak['day_name']} at {int(peak['hour']):02d}:00")
print(f"    -> Recommend {int(peak['recommended_drivers'])} drivers")
print(f"    -> Expect {int(peak['confidence_low'])}–{int(peak['confidence_high'])} trips (90% CI)")
print()
print(f"  Quietest slot: {quiet['day_name']} at {int(quiet['hour']):02d}:00")
print(f"    -> Recommend {int(quiet['recommended_drivers'])} drivers")
print()
print(f"  Model error: ~{cv_mae.mean():.1f} trips off on average")
print()
print(f"  Output files saved to: ./{OUTPUT_FOLDER}/")
print("    1_eda_charts.png")
print("    2_statistical_analysis.png")
print("    3_demand_forecast.png")
print("    4_ml_results.png")
print("    driver_recommendations.csv")
print("="*55)
print("\nDone!")
