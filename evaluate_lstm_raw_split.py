"""Out-of-sample evaluation of the saved LSTM on the notebook's own split.

The previous version of this script split the raw 2,014-row closing-price
series 80/20, which put its first test date at 2021-05-27. The notebook's
training slice ends on 2021-06-10, so the first ten dates of that test window
had already been seen by both the saved model and the saved scaler. They were
scored as if they were out of sample, and they pulled the reported RMSE down.

This version derives the boundary from the saved artifacts instead of
inventing a second split:

  * the notebook's feature frame has 1,964 rows after the indicator warm-up
    (Cell 15 output: "After dropna: (1964, 13)"), so the raw series loses its
    first 50 rows;
  * ``scaler.n_samples_seen_`` is 1,571, which must equal
    ``int(1,964 * 0.8)`` -- i.e. the notebook's own training fraction;
  * the test window is everything strictly after the last training date.

Nothing is retrained, no model file is written, and the 403-row result
produced by the previous version is left untouched on disk.
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
data_path = Path("data/AAPL.csv")
model_path = Path("models/lstm_model.keras")
scaler_path = Path("models/scaler.pkl")
output_path = Path("outputs/lstm_out_of_sample_evaluation.csv")

# Recorded by the notebook. Cell 15 prints "After dropna: (1964, 13)" and
# Cell 35 splits ``df`` at ``int(len(df) * 0.8)``.
NOTEBOOK_FRAME_ROWS = 1964
NOTEBOOK_SPLIT_FRACTION = 0.8
look_back = 60

# Load and clean the cached Yahoo Finance CSV. The committed file is the
# reproducible source; nothing here writes to data/.
raw = pd.read_csv(data_path)
raw["Price"] = pd.to_datetime(raw["Price"], errors="coerce", format="mixed")
raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")

clean = (
    raw.dropna(subset=["Price", "Close"])
       .sort_values("Price")
       .drop_duplicates(subset=["Price"])
       .reset_index(drop=True)
)

close = clean[["Close"]]
dates = clean["Price"]

# Load existing artifacts; do not retrain, refit or save them.
scaler = joblib.load(scaler_path)
model = tf.keras.models.load_model(model_path, compile=False)

# --- Boundary derived from the saved scaler, then checked against the notebook.
n_train = int(scaler.n_samples_seen_)

# The notebook computes its indicators on the raw series and drops the warm-up
# rows (the longest window is MA_50, so 50 rows go), leaving a 1,964-row frame.
warmup = len(close) - NOTEBOOK_FRAME_ROWS
if warmup <= 0:
    raise ValueError(
        f"Committed series has {len(close)} rows, which cannot contain the "
        f"notebook's {NOTEBOOK_FRAME_ROWS}-row feature frame."
    )

frame = close.iloc[warmup:].reset_index(drop=True)
if len(frame) != NOTEBOOK_FRAME_ROWS:
    raise ValueError(
        f"Reconstructed frame has {len(frame)} rows, expected "
        f"{NOTEBOOK_FRAME_ROWS}; the committed data no longer matches the notebook."
    )

expected_train = int(NOTEBOOK_FRAME_ROWS * NOTEBOOK_SPLIT_FRACTION)
if n_train != expected_train:
    raise ValueError(
        f"Saved scaler was fitted on {n_train} rows, but the notebook's "
        f"{NOTEBOOK_SPLIT_FRACTION:.0%} split of {NOTEBOOK_FRAME_ROWS} rows gives "
        f"{expected_train}. The scaler does not belong to this split."
    )

train = frame.iloc[:n_train]
test = frame.iloc[n_train:]

train_dates = dates.iloc[warmup:warmup + n_train].reset_index(drop=True)
test_dates = dates.iloc[warmup + n_train:].reset_index(drop=True)

# The scaler's min/max are a weak check on their own -- the series' extremes sit
# far inside both candidate windows -- so `n_samples_seen_` above is the real
# test and this one only confirms the window's bounds.
if not np.isclose(train["Close"].min(), scaler.data_min_[0]):
    raise ValueError("Training minimum does not match the saved scaler.")
if not np.isclose(train["Close"].max(), scaler.data_max_[0]):
    raise ValueError("Training maximum does not match the saved scaler.")

if test_dates.iloc[0] <= train_dates.iloc[-1]:
    raise ValueError(
        f"Test window starts on {test_dates.iloc[0]:%Y-%m-%d}, which is not "
        f"strictly after the last training date {train_dates.iloc[-1]:%Y-%m-%d}."
    )

scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# Include the last 60 training observations as context for the first test day.
sequence_data = np.concatenate([scaled_train[-look_back:], scaled_test], axis=0)

X_test = np.array([
    sequence_data[i - look_back:i, 0]
    for i in range(look_back, len(sequence_data))
]).reshape(-1, look_back, 1)

actual = test["Close"].to_numpy()
baseline = frame["Close"].iloc[n_train - 1:-1].to_numpy()

predicted_scaled = model.predict(X_test, verbose=0)
predicted = scaler.inverse_transform(predicted_scaled).ravel()

if not (len(actual) == len(predicted) == len(baseline) == len(test_dates)):
    raise ValueError(
        f"Misaligned arrays: actual={len(actual)}, LSTM={len(predicted)}, "
        f"baseline={len(baseline)}, dates={len(test_dates)}"
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

# One row per scored date, date-aligned with the actual close.
predictions = pd.DataFrame({
    "Date": test_dates.to_numpy(),
    "actual_close": actual,
    "lstm_prediction": predicted,
    "previous_close_baseline": baseline,
})

output_path.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(output_path, index=False)

print("Out-of-sample LSTM evaluation")
print("-" * 46)
print(f"Committed series rows:     {len(close)}")
print(f"Indicator warm-up dropped: {warmup} rows (notebook frame: {len(frame)} rows)")
print(
    f"Training slice:            {n_train} rows, "
    f"{train_dates.iloc[0]:%Y-%m-%d} to {train_dates.iloc[-1]:%Y-%m-%d}"
)
print(
    f"Test slice:                {len(test)} rows, "
    f"{test_dates.iloc[0]:%Y-%m-%d} to {test_dates.iloc[-1]:%Y-%m-%d}"
)
print(f"Scaler n_samples_seen_:    {n_train} (matches int({len(frame)} * 0.8))")
print("Boundary check:            test starts strictly after the last training date")
print("\nMetrics:")
print(comparison.to_string(index=False))
print("\nSaved new predictions to:", output_path)
print("outputs/lstm_raw_split_evaluation.csv is left unchanged for provenance;")
print("its first 10 rows are in-sample and it is superseded by this file.")
