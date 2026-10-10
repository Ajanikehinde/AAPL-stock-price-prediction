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

The project uses an **80/20 chronological train-test split** without shuffling.

---

## Key Finding

The most important finding was related to **data leakage**.

The original implementation calculated technical indicators directly from the current closing price. This allowed information from the prediction target to influence the input features.

The corrected implementation shifts the closing price before calculating the indicators so that each feature uses only information available before the prediction point.

The intentionally leaky model produced an almost perfect result:

```text
RMSE approximately 7.97 x 10^-14
```

because the target `Close` was explicitly included among the leaky features.

After correcting the leakage:

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
| Random Forest - returns     |  3.6662 |  2.9310 |  1.9455% |
| ARIMA                       | 22.1046 | 18.4771 | 11.8274% |

The naive baseline is included deliberately. Without a baseline, a model can appear successful simply because historical stock prices are strongly autocorrelated.

**Evaluation note:** The table above covers the feature-engineered model comparison. The LSTM row has been omitted because the existing comparison CSV assigned its predictions incorrect dates. The saved LSTM was evaluated separately on the cleaned raw closing-price series, with correctly aligned dates and the same test observations for both the LSTM and previous-close baseline. The separate results are reported below.

---

## Separate LSTM Evaluation

The saved LSTM model was evaluated on the cleaned raw AAPL closing-price series using an 80/20 chronological split. The saved scaler's minimum and maximum matched the raw training-period closing prices. The evaluation uses a 60-day lookback window, with the final 60 training observations providing context for the first test prediction.

The evaluation covers **403 test observations from 27 May 2021 to 30 December 2022**.

| Model | RMSE | MAE | MAPE |
| --- | ---: | ---: | ---: |
| Previous-close baseline | 2.9258 | 2.2061 | 1.4780% |
| Saved LSTM | 5.9547 | 4.7899 | 3.1752% |

The saved LSTM did not outperform the previous-close baseline on any of these metrics. This evaluates the existing saved artifact; it does not establish that every LSTM configuration will perform similarly. The evaluation script is `evaluate_lstm_raw_split.py`, and its date-aligned predictions are in `outputs/lstm_raw_split_evaluation.csv`.

This raw-price evaluation is separate from the feature-engineered comparison above. The results should not be combined into a single model ranking without matching the target definitions, dates, and evaluation procedure.

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
Maximum reconstructed prediction: $174.55
Actual test-period maximum:       $177.79
```

Its reconstructed-price RMSE was **3.6662**, which was still higher than the naive baseline.

This return-based experiment is treated separately because it predicts the **next day's return and reconstructs the next day's closing price**, rather than using exactly the same target alignment as the main price-level models.

---

## Methodology

### 1. Data acquisition

Historical AAPL data is obtained using `yfinance`.

The notebook also supports loading a cached CSV from:

```text
data/AAPL.csv
```

This provides a fallback when live data cannot be downloaded.

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

A separate Random Forest experiment predicts percentage returns to address the inability of tree-based models to extrapolate beyond training target values.

### 6. ARIMA

`pmdarima.auto_arima` is used to automatically select a non-seasonal ARIMA configuration.

The selected model was:

```text
SARIMAX(2, 1, 3)
```

The model is fitted using a positional index to avoid datetime-frequency issues caused by weekends and market holidays.

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

---

## Leakage Prevention

Two important leakage problems from the original implementation were addressed.

### Technical-indicator leakage

Indicators were originally calculated directly from `Close`.

The corrected implementation first creates:

```python
shifted_close = df['Close'].shift(1)
```

and calculates the indicators from that historical series.

### Scaling leakage

The original LSTM implementation fitted the scaler on the entire dataset.

The corrected implementation fits the scaler only on the training period:

```python
scaler.fit_transform(train_close_vals)
scaler.transform(test_close_vals)
```

This prevents test-period information from influencing the training transformation.

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
|-- lstm_raw_split_evaluation.csv
```

`predictions_comparison.csv` contains the existing predictions for the feature-engineered comparison. Its LSTM dates are not considered reliable for evaluation and its LSTM row is therefore excluded from the performance table above.

`rf_returns_predictions.csv` contains the separately aligned next-day predictions from the return-based Random Forest experiment.

`lstm_raw_split_evaluation.csv` contains date-aligned predictions from the separate raw-price LSTM evaluation, including actual closing prices, LSTM predictions, and the previous-close baseline.

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
* The return-based Random Forest uses a different prediction alignment from the main price-level models.
* The LSTM uses the final 60 training observations as context for the beginning of the test sequence, which is standard for a windowed time-series model.
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

The notebook downloads AAPL stock data using `yfinance` when available and
can fall back to the included `data/AAPL.csv` file.

## Conclusion

This project demonstrates the importance of leakage detection and realistic baselines in financial machine learning.

The corrected experiments show that removing leakage substantially changes the apparent performance of the forecasting model. Linear Regression performs very close to the previous-close baseline in the feature-engineered comparison. The price-level Random Forest and ARIMA also fail to demonstrate an advantage there. In the separate raw-price evaluation, the saved LSTM performs worse than the previous-close baseline on RMSE, MAE, and MAPE.

The main takeaway is methodological: **a strong-looking prediction result is not sufficient evidence of predictive skill unless the feature construction, temporal validation, leakage controls, and naive baseline are all handled correctly.**
