import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths
data_path = Path("data/AAPL.csv")
model_path = Path("models/lstm_model.keras")
scaler_path = Path("models/scaler.pkl")
output_path = Path("outputs/lstm_raw_split_evaluation.csv")

# Load and clean the cached Yahoo Finance CSV.
raw = pd.read_csv(data_path)
raw["Price"] = pd.to_datetime(raw["Price"], errors="coerce")
raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")

clean = (
    raw.dropna(subset=["Price", "Close"])
       .sort_values("Price")
       .drop_duplicates(subset=["Price"])
       .reset_index(drop=True)
)

close = clean[["Close"]]
look_back = 60
split = int(len(close) * 0.8)

train = close.iloc[:split]
test = close.iloc[split:]

# Load existing artifacts; do not retrain or save them.
scaler = joblib.load(scaler_path)
model = tf.keras.models.load_model(model_path, compile=False)

# Check that this split matches the scaler's fitted training range.
if not np.isclose(train["Close"].min(), scaler.data_min_[0]):
    raise ValueError("Training minimum does not match the saved scaler.")
if not np.isclose(train["Close"].max(), scaler.data_max_[0]):
    raise ValueError("Training maximum does not match the saved scaler.")

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Include the last 60 training observations as context for the first test day.
sequence_data = np.concatenate(
    [scaled_train[-look_back:], scaled_test], axis=0
)

X_test = np.array([
    sequence_data[i-look_back:i, 0]
    for i in range(look_back, len(sequence_data))
]).reshape(-1, look_back, 1)

actual = test["Close"].to_numpy()
baseline = close["Close"].iloc[split-1:-1].to_numpy()

predicted_scaled = model.predict(X_test, verbose=0)
predicted = scaler.inverse_transform(predicted_scaled).ravel()

if not (len(actual) == len(predicted) == len(baseline)):
    raise ValueError(
        f"Misaligned arrays: actual={len(actual)}, "
        f"LSTM={len(predicted)}, baseline={len(baseline)}"
    )

def metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }

comparison = pd.DataFrame([
    {"Model": "Saved LSTM", **metrics(actual, predicted)},
    {"Model": "Previous-close baseline", **metrics(actual, baseline)},
]).round(4)

# Preserve the actual date for every prediction.
predictions = pd.DataFrame({
    "Date": clean["Price"].iloc[split:].to_numpy(),
    "actual_close": actual,
    "lstm_prediction": predicted,
    "previous_close_baseline": baseline,
})

output_path.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(output_path, index=False)

print("Evaluation complete.")
print("Clean rows:", len(close))
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Test period:", clean["Price"].iloc[split], "to", clean["Price"].iloc[-1])
print("\nMetrics:")
print(comparison.to_string(index=False))
print("\nSaved new predictions to:", output_path)
print("Existing model files and original comparison were not modified.")
