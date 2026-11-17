# AAPL Stock Price Prediction

A time-series machine learning project comparing Linear Regression, Random Forest, ARIMA, and LSTM models for predicting Apple (AAPL) closing prices using historical market data and technical indicators.

The project focuses not only on model performance, but also on identifying and correcting data leakage, maintaining time-aware validation, and comparing machine-learning models against a simple naive baseline.

---

## Project Overview

This project uses AAPL historical price data from **2015-2022** to investigate whether technical indicators and historical price information can provide reliable next-day forecasting performance.

The notebook evaluates:

* Linear Regression
* Random Forest
* Random Forest trained on returns
* ARIMA
* LSTM
* Naive previous-close baseline

Technical indicators include:

* 20-day moving average
* 50-day moving average
* 20-day rolling volatility
* 14-day RSI
* MACD
* MACD signal
* Lagged closing prices

Each experiment uses an **80/20 chronological train-test split** without shuffling; the exact split differs by experiment (see Methodology).

---

## Key Finding

The most important finding was related to **data leakage**.

The original implementation calculated technical indicators directly from the current closing price. This allowed information from the prediction target to influence the input features.

The corrected implementation shifts the closing price before calculating the indicators so that each feature uses only information available before the prediction point.

The notebook's deliberately leaky demonstration produced an almost perfect result:

```text
RMSE approximately 7.97 x 10^-14
```

That number has a narrow meaning. The demonstration also adds the target `Close` **itself** to the feature matrix (`features_leaky` in notebook Cell 23), so the linear model learns the identity mapping `Close -> Close`. It shows that target leakage produces a meaningless near-zero error; it does **not** quantify the error caused by the unshifted indicators alone.

A separate, earlier variant used only the unshifted indicators (no `Close` column) and scored RMSE **1.3937**, compared with **2.9938** for the corrected features on the same run. That variant is not reproduced by the committed notebook cells, so it is reported as provenance rather than as a saved result.

After correcting the leakage, on the notebook's 393-date test window:

```text
Naive baseline RMSE:       2.9573
Linear Regression RMSE:    2.9857
```

The corrected Linear Regression model therefore did not demonstrate an improvement over simply using the previous day's closing price.

This result is consistent with the difficulty of extracting reliable next-day price information from historical prices alone.

---

## Model Performance

| Model                       |    RMSE |     MAE |     MAPE |
| --------------------------- | ------: | ------: | -------: |
| Naive - previous close      |  2.9573 |  2.2387 |  1.4964% |
| Linear Regression           |  2.9857 |  2.2707 |  1.5177% |
| Random Forest - price level | 18.6093 | 14.7567 |  9.3060% |
| Random Forest - returns     |  3.6784 |  2.9413 |  1.9524% |
| ARIMA (fixed-origin)        | 23.9167 | 20.4237 | 13.0756% |
| LSTM (notebook split)       |  6.0268 |  4.8836 |  3.2330% |

The naive baseline is included deliberately. Without a baseline, a model can appear successful simply because historical stock prices are strongly autocorrelated.

**Evaluation note.** The naive, Linear Regression, Random Forest price-level, ARIMA and LSTM rows are recomputed from `outputs/predictions_comparison.csv`; the Random Forest returns row is recomputed from `outputs/rf_returns_predictions.csv`. All six match the notebook's own comparison table (Cell 40). The LSTM row is the notebook's 393-date test window (2021-06-11 to 2022-12-30), the same window as the other rows. A second evaluation of the saved LSTM, feeding it only the closing-price sequence, is reported separately below; it scores the same window and agrees with this row.

---

## Separate LSTM Evaluation

`evaluate_lstm_raw_split.py` re-evaluates the saved LSTM artifact on the cleaned closing-price series, using the notebook's own split rather than a second ad-hoc one. It drops the 50-row indicator warm-up (the longest window is `MA_50`) to reach the notebook's 1,964-row feature frame, reads the boundary from the saved scaler's `n_samples_seen_` (1,571, which must equal `int(1964 * 0.8)`), and refuses to score any date at or before the last training date.

Verified boundary:

```text
Training slice: 1,571 rows, 2015-03-17 to 2021-06-10
Test slice:       393 rows, 2021-06-11 to 2022-12-30
```

| Model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| Previous-close baseline | 2.9573 | 2.2387 | 1.4964% |
| Saved LSTM | 6.0268 | 4.8836 | 3.2330% |

These are the same 393 dates, the same target definition and the same metrics as the notebook's own LSTM row, which is why the two agree exactly. The saved LSTM does not outperform the previous-close baseline on any of the three metrics. Its dated predictions are in `outputs/lstm_out_of_sample_evaluation.csv`.

**Superseded result.** An earlier version of this script split the raw 2,014-row series 80/20 instead, so its test window began 2021-05-27 — ten trading days before the scaler's fitting window and the model's training targets ended (2021-06-10). Those ten dates were in-sample and scored RMSE 1.2387, pulling the reported 403-date RMSE down to **5.9547** (MAE 4.7899, MAPE 3.1752%). That file, `outputs/lstm_raw_split_evaluation.csv`, is kept unchanged for provenance and should not be quoted as an out-of-sample result.

This evaluation shares the main comparison's target (the next-day close) but not its input representation: here the LSTM consumes a univariate 60-day closing-price sequence, with no indicator frame involved. The two sections should not be merged into one ranking without matching target definitions, dates and procedure.

---

## Random Forest: Price-Level vs Returns

The price-level Random Forest performed poorly because tree-based models cannot extrapolate beyond the target values observed during training.

In this experiment:

```text
Highest training Close:       $138.98
Maximum RF prediction:        $136.51
Highest test-period Close:    $177.79
```

There were **323 of 393 test observations** above the maximum closing price observed during training.

The experiment was therefore reformulated so that Random Forest predicted the next-day percentage return instead of the absolute price.

The return-based model could produce predictions above the training-period price ceiling:

```text
Maximum reconstructed prediction: $174.54
Actual test-period maximum:       $177.79
```

Its reconstructed-price RMSE was **3.6784** (MAE 2.9413, MAPE 1.9524%). That is still worse than the previous-close baseline on the same target dates, whose RMSE is **2.9611** (MAE 2.2436, MAPE 1.4996%) over the 392 dates the return model covers. Comparing 3.6784 to the 393-date one-step baseline (2.9573) mixes windows; the conclusion is unchanged either way.

Two alignment details matter:

* `outputs/rf_returns_predictions.csv` names both dates explicitly: `feature_date` is the day `t` whose close feeds the reconstruction, and `target_date` is the following trading day being predicted. Its first row pairs 2021-06-10 with 2021-06-11; its last pairs 2022-12-28 with 2022-12-29. The earlier version labelled the single date column `Date`, which read like a prediction date.
* The return model covers 392 of the 393 targets used by the price-level models. The final row of `df_ret` is dropped before `shift(-1)`, so the 2022-12-30 target is never evaluated.

The file also stores `previous_close_baseline` for every row — literally `Close[t]`, the naive forecast for `target_date` — so the 2.9611 baseline above is computed on the identical 392 rows rather than borrowed from the 393-date one-step table.

Features at `t` never include `Close[t]` (all indicators are built on `Close.shift(1)`), and the reconstruction uses the observed `Close[t]`, which is legitimate for a forecast made at the close of day `t`. The return model uses fixed hyperparameters (`n_estimators=200`) and is not tuned by `GridSearchCV` like the price-level model, so the two Random Forest rows are not a like-for-like architecture comparison.

This return-based experiment is treated separately because it predicts the **next day's return and reconstructs the next day's closing price**, rather than using exactly the same target alignment as the main price-level models.

---

## Results Provenance

Every number reported above is either recomputed from a committed output file or quoted from a specific notebook cell. Cell numbers count every cell in the notebook (markdown and code). The notebook is `AAPL_Prediction_corrected.ipynb`; the CSVs are in `outputs/`.

| Result | Value | Committed source |
| --- | ---: | --- |
| Naive previous-close baseline | 2.9573 | Cell 21 / Cell 40; `predictions_comparison.csv` (`pred_naive`) |
| Linear Regression | 2.9857 | Cell 21 / Cell 40; `pred_lr` |
| Random Forest, price level | 18.6093 | Cell 26 / Cell 40; `pred_rf_price`; `models/rf_model.pkl` |
| Random Forest, returns | 3.6784 | Cell 30 / Cell 40; `rf_returns_predictions.csv` (written by Cell 45) |
| ARIMA, fixed-origin | 23.9167 | Cell 33 / Cell 40; `pred_arima` |
| LSTM, notebook split | 6.0268 | Cell 37 / Cell 40; `pred_lstm` |
| LSTM, out-of-sample script | 6.0268 | `evaluate_lstm_raw_split.py`; `lstm_out_of_sample_evaluation.csv` |
| LSTM, superseded 403-date split | 5.9547 | `lstm_raw_split_evaluation.csv`; 10 of its rows are in-sample |
| Leaky demo (with `Close` feature) | 7.97e-14 | Cell 23 |
| Leaky, unshifted indicators only | 1.3937 | Earlier run only; **not** in the committed notebook |

Status of each claim:

* **Implemented, with saved predictions:** the six notebook rows, the leaky demonstration, the price-level Random Forest artifact, the scaler and the LSTM artifact. These can be checked from the CSVs and model files.
* **Implemented but not saved as a fitted model:** Linear Regression, ARIMA and the return-based Random Forest. Their predictions are saved, but the estimators are not.
* **Referenced but not reproducible from committed code:** the 1.3937 unshifted-indicator-only result. It comes from an earlier revision and is included only to explain the "1.39 to 2.99" wording in the notebook's introduction.
* **Assumptions:** the previous-close baseline is the correct one-step naive comparator; ARIMA's fixed-origin protocol is intentional; `evaluate_lstm_raw_split.py` treats `scaler.n_samples_seen_` as the split boundary and uses the scaler's min/max only as a bounds check, since the series' extremes fall well inside both candidate windows.

**Correction note.** Earlier drafts of this README quoted Random Forest returns at 3.6662 and ARIMA at 22.1046. Neither value is produced by the committed notebook or any committed output. The committed values are 3.6784 and 23.9167, and they are the values used here.

---

## Methodology

### 1. Data acquisition

The committed file is the default source:

```text
data/AAPL.csv
```

`load_stock_data()` reads that file and every number in this README was produced from it. A live `yfinance` download is opt-in (`download=True`), and even then it stays in memory unless `overwrite=True` is passed explicitly. Earlier versions downloaded first and wrote `df.to_csv(path)` on every run, silently replacing the tracked dataset; see Reproducibility for why that changed the metrics.

### 2. Feature engineering

The following features are calculated using historical information:

```text
MA_20
MA_50
Vol_20
RSI_14
MACD
MACD_signal
Lag_1
Lag_2
```

The technical indicators are calculated using a shifted closing-price series to avoid using the current target value when constructing predictors.

### 3. Train-test split

The dataset is split chronologically:

```text
Training: 2015-03-17 to 2021-06-10
Testing:  2021-06-11 to 2022-12-30
```

There is no random shuffling.

Two frames appear in the project: the main feature-engineered frame (1,964 rows) and the leaky-demo frame (1,915 rows, because re-running the 50-day rolling window drops 49 more warm-up rows). `evaluate_lstm_raw_split.py` scores the same 393 dates as the main frame, after dropping the 50-row indicator warm-up from the raw 2,014-row series.

### 4. Linear Regression

Linear Regression provides a simple machine-learning baseline using the engineered technical indicators and lagged prices.

### 5. Random Forest

A Random Forest Regressor is tuned using `GridSearchCV` with `TimeSeriesSplit`.

Best parameters for the price-level model:

```text
n_estimators:       100
max_depth:          10
min_samples_split:  5
```

A separate Random Forest experiment predicts percentage returns to address the inability of tree-based models to extrapolate beyond training target values. That model uses fixed hyperparameters (`n_estimators=200`, `random_state=42`) and is not tuned; the price-level parameters above do not transfer to it.

### 6. ARIMA

`pmdarima.auto_arima` is used to automatically select a non-seasonal ARIMA configuration.

The selected model was:

```text
SARIMAX(2, 1, 3)
```

The model is fitted using a positional index to avoid datetime-frequency issues caused by weekends and market holidays.

**Evaluation caveat.** The ARIMA forecast is generated once from the end of the training set: `predict(n_periods=len(y_test))` returns a 393-step-ahead path from a single origin. No observed test-period price is fed back. Linear Regression, Random Forest, the LSTM and the naive baseline are all effectively one-step-ahead, so the ARIMA RMSE is not a like-for-like comparison. Under a matching fixed-origin benchmark (repeat the last training close, 122.82), ARIMA's 23.9167 does beat the naive 29.8879; it is the one-step baseline (2.9573) it cannot match. A rolling-origin, refitted one-step ARIMA is not implemented.

### 7. LSTM

The LSTM uses a 60-day lookback window.

The `MinMaxScaler` is fitted **only on the training data** and then applied to the test data. This prevents future test-period price levels from influencing the training transformation.

The model contains:

* LSTM layer: 150 units
* LSTM layer: 100 units
* Dropout: 0.3
* Dense output layer
* Adam optimizer
* Early stopping

**Validation.** `validation_split=0.1` holds out the last 10% of the training sequences before shuffling, so early stopping never sees the test period. Because consecutive 60-day windows overlap by 59 steps, the validation sequences are highly correlated with the training sequences and `val_loss` is an optimistic, non-independent signal. The held-out block is also the training data closest to the test period, so model selection leans toward the most recent regime.

**Artifact/split note.** `models/scaler.pkl` reports `n_samples_seen_ = 1571`, confirming it was fitted on the notebook's 1,571-row training slice (2015-03-17 to 2021-06-10). The separate `evaluate_lstm_raw_split.py` derives the same boundary from that count as `int(1964 * 0.8)`, so it scores only dates after 2021-06-10; see the LSTM evaluation section above.

---

## Leakage Prevention

Two leakage problems from the original implementation were addressed in the notebook. A third problem — an evaluation window that reached back into the training period — was present in the separate LSTM script and has now been corrected there.

### Technical-indicator leakage

Indicators were originally calculated directly from `Close`.

The corrected implementation first creates:

```python
shifted_close = df['Close'].shift(1)
```

and calculates the indicators from that historical series.

The notebook's Section 6a demonstration of the bug also adds `Close` itself to the feature list, so its ~8e-14 RMSE is a target-as-input identity, not a measurement of indicator leakage alone. The narrower unshifted-indicators-only variant scored 1.3937 in an earlier run (versus 2.9938 corrected), but that variant is not part of the committed notebook.

### Scaling leakage

The original LSTM implementation fitted the scaler on the entire dataset.

The corrected implementation fits the scaler only on the training period:

```python
scaler.fit_transform(train_close_vals)
scaler.transform(test_close_vals)
```

This prevents test-period information from influencing the training transformation.

### Boundary control in the separate LSTM script

An earlier version of `evaluate_lstm_raw_split.py` split the raw 2,014-row series 80/20, so its test window began 2021-05-27 — ten trading days before the notebook's training slice ends. The saved scaler was fitted through 2021-06-10 and the model's training targets end on the same date, so those ten dates were in-sample; they scored RMSE 1.2387 and pulled the reported 403-date RMSE down to 5.9547.

The script now drops the 50-row indicator warm-up, asserts `scaler.n_samples_seen_ == int(1964 * 0.8)`, and refuses to score any date at or before the last training date. It outputs 393 dates (2021-06-11 to 2022-12-30) with RMSE 6.0268, matching the notebook exactly. The overlap was a property of the evaluation script, not of the notebook's training loop.

---

## Explainability

SHAP is used to examine feature contributions for the tuned price-level Random Forest.

The SHAP analysis is specifically applied to the Random Forest model and should not be interpreted as an explanation of the ARIMA, Linear Regression, or LSTM models.

---

## Outputs

The project saves trained models to:

```text
models/
|-- rf_model.pkl
|-- scaler.pkl
|-- lstm_model.keras
```

Prediction outputs are saved to:

```text
outputs/
|-- predictions_comparison.csv
|-- rf_returns_predictions.csv
|-- lstm_out_of_sample_evaluation.csv
|-- lstm_raw_split_evaluation.csv   (superseded, kept for provenance)
```

`predictions_comparison.csv` contains the predictions for the feature-engineered comparison, including the LSTM predictions on the notebook's 393-date test window. Its LSTM column is correctly aligned with `y_test`: recomputing its metrics gives 6.0268 / 4.8836 / 3.2330%, matching the notebook's Cell 37 and Cell 40 outputs. (An earlier draft of this README claimed the LSTM dates were unreliable; that claim was incorrect and has been removed.)

`rf_returns_predictions.csv` contains the separately aligned next-day predictions from the return-based Random Forest experiment. It writes `feature_date` (`t`) and `target_date` (`t+1`) as separate columns, along with `actual_next_close`, `pred_rf_returns` and `previous_close_baseline` (`Close[t]`), covering 392 target dates (2021-06-11 to 2022-12-29).

`lstm_out_of_sample_evaluation.csv` contains the corrected evaluation of the saved LSTM: 393 rows from 2021-06-11 to 2022-12-30, with the actual close, the LSTM prediction and the previous-close baseline. Every row is strictly out of sample.

`lstm_raw_split_evaluation.csv` is the superseded 403-row output from the earlier 80/20 raw-series split (2021-05-27 to 2022-12-30). Its first 10 rows are in-sample with respect to the saved model, so it is retained only to document the correction; use `lstm_out_of_sample_evaluation.csv` instead.

Only three artifacts are persisted: `models/rf_model.pkl` (the tuned price-level Random Forest), `models/scaler.pkl` and `models/lstm_model.keras`. The Linear Regression, ARIMA and return-based Random Forest models are not saved; their predictions can be inspected in the CSVs, but reproducing them requires re-running the notebook.

## Project Structure

```text
AAPL-stock-price-prediction/
|-- AAPL_Prediction_corrected.ipynb
|-- evaluate_lstm_raw_split.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- AAPL.csv
|-- models/
|   |-- rf_model.pkl
|   |-- scaler.pkl
|   |-- lstm_model.keras
|-- outputs/
|   |-- predictions_comparison.csv
|   |-- rf_returns_predictions.csv
|   |-- lstm_out_of_sample_evaluation.csv
|   |-- lstm_raw_split_evaluation.csv
```

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* scikit-learn
* TensorFlow / Keras
* pmdarima
* SHAP
* pandas-ta
* yfinance
* Joblib
* Jupyter Notebook

---

## Limitations

This project should be interpreted as an experimental forecasting study rather than a trading system.

Key limitations include:

* No transaction costs or slippage are modeled.
* No position-sizing or portfolio-management strategy is evaluated.
* Only one stock is studied.
* Only the 2015-2022 period is evaluated.
* The models are not tested across multiple market regimes or additional tickers.
* The models are not all evaluated under the same task. ARIMA is a single fixed-origin 393-step forecast; the other models are effectively one-step-ahead and use observed test-period prices. ARIMA's much larger RMSE is partly a task-definition difference.
* The return-based Random Forest predicts next-day returns, reconstructs price from the observed current close, covers 392 of 393 target dates, and is not tuned like the price-level Random Forest.
* The separate LSTM evaluation was corrected after the fact: its superseded 403-date split included 10 in-sample dates and reported an optimistic 5.9547. The current script scores only out-of-sample dates and reports 6.0268 on 393 dates.
* The LSTM uses the final 60 training observations as context for the beginning of the test sequence, which is standard for a windowed time-series model.
* The LSTM validation split overlaps heavily with training windows (59 of 60 steps), so early-stopping validation is not independent.
* The leakage demonstration includes the target `Close` as an input; its near-zero error is a degenerate identity, not an isolated measure of indicator leakage.
* Two different frames are used (1,964 rows main, 1,915 leaky demo), so metrics across sections are not directly comparable without matching the split; the separate LSTM evaluation scores the same 393 dates as the main frame.
* No claim of profitable trading performance is made.

---

## Reproducibility

The notebook uses a fixed random seed (`SEED = 42`). It creates the required
`data/`, `models/`, and `outputs/` directories as needed.

The dependencies below are pinned to the versions used in the Python 3.12
environment for this project.

To clone the repository and run the notebook on Windows:

```powershell
git clone https://github.com/Ajanikehinde/AAPL-stock-price-prediction.git
cd AAPL-stock-price-prediction
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook AAPL_Prediction_corrected.ipynb
```

The notebook loads the committed `data/AAPL.csv` by default and never writes to it. A live download is opt-in (`download=True`) and persists only when `overwrite=True` is also passed, so running the notebook can no longer replace the tracked dataset.

**Use the committed data file for exact reproduction.** With `auto_adjust=True`, historical prices are back-adjusted for every dividend up to the download date, so a fresh download does not match the committed file. Checked on 2026-11-16, a fresh 2015-2022 download differed on all 2,014 rows (mean absolute difference 0.053, maximum 0.135). Re-running against live data can therefore change the metrics and even the `auto_arima` order: an earlier run (preserved in a local, uncommitted checkpoint) selected `SARIMAX(2,1,2)` and produced ARIMA RMSE 21.1046, while the committed run selected `SARIMAX(2,1,3)` and produced 23.9167.

The notebook's random seeds (`SEED = 42`) cover NumPy, Python's `random` and TensorFlow, and the saved Random Forest artifact loads with the recorded `random_state=42`, `n_estimators=100`, `max_depth=10` and `min_samples_split=5`. ARIMA's `auto_arima` search is not seeded and is data-dependent. TensorFlow results can also vary in the last decimal across hardware and library versions.

## Conclusion

This project demonstrates the importance of leakage detection and realistic baselines in financial machine learning.

The corrected experiments show that removing leakage substantially changes the apparent performance of the forecasting model. Linear Regression performs very close to the previous-close baseline in the feature-engineered comparison (2.9857 vs 2.9573 RMSE). The price-level Random Forest (18.6093) and the return-based Random Forest (3.6784 vs a 2.9611 comparable baseline) do not demonstrate an advantage. ARIMA's 23.9167 comes from a much harder, fixed-origin multi-step task and should not be read as a like-for-like one-step ranking, although it does beat a matching fixed-origin naive benchmark (29.8879). The saved LSTM loses to the previous-close baseline on RMSE, MAE and MAPE; once its evaluation window is aligned with the notebook's split, its out-of-sample RMSE is 6.0268 versus 2.9573.

The main takeaway is methodological: **a strong-looking prediction result is not sufficient evidence of predictive skill unless the feature construction, temporal validation, leakage controls, evaluation protocol and naive baseline are all handled correctly.**
