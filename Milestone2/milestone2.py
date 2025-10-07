# forecasting_all_products.py - Milestone 2 (Improved for per-product forecasting)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pickle
from math import pi
import warnings
warnings.filterwarnings("ignore")


# ======================= Utility Functions =======================

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE safely (avoiding division by zero)."""
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def calculate_metrics(y_true, y_pred, model_name):
    """Calculates MAE, RMSE, MAPE and returns as dictionary."""
    y_pred[y_pred < 0] = 0
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'Model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def prepare_data(df_raw, split_date='2023-12-01'):
    """Splits the given dataframe into train and test sets."""
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_agg = df_raw.groupby('Date')['Units Sold'].sum().reset_index()

    split_index = df_agg[df_agg['Date'].dt.date >= pd.to_datetime(split_date).date()].index.min()
    train_data = df_agg.iloc[:split_index].copy()
    test_data = df_agg.iloc[split_index:].copy()
    y_test_true = test_data['Units Sold'].values
    return train_data, test_data, y_test_true


# ======================= Forecasting Models =======================

def run_prophet(train_data, test_data):
    """Runs Prophet model and returns metrics + forecast."""
    df_prophet = train_data.rename(columns={'Date': 'ds', 'Units Sold': 'y'})
    m = Prophet(
        seasonality_mode='multiplicative',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=len(test_data), include_history=False)
    forecast = m.predict(future)
    y_pred = forecast['yhat'].values
    return calculate_metrics(test_data['Units Sold'].values, y_pred, 'Prophet'), m, y_pred


def run_lstm(train_data, test_data, time_step=30):
    """Runs LSTM model for total store sales."""
    sales_data = train_data['Units Sold'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(sales_data)

    X_train, Y_train = [], []
    for i in range(len(scaled_data) - time_step):
        X_train.append(scaled_data[i:(i + time_step), 0])
        Y_train.append(scaled_data[i + time_step, 0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=20, batch_size=64, verbose=0)

    last_train_window = scaler.transform(train_data['Units Sold'].iloc[-time_step:].values.reshape(-1, 1)).flatten()
    Y_pred_scaled = []
    current_batch = last_train_window
    for _ in range(len(test_data)):
        current_pred_scaled = model.predict(current_batch.reshape(1, time_step, 1), verbose=0)[0]
        Y_pred_scaled.append(current_pred_scaled)
        current_batch = np.append(current_batch[1:], current_pred_scaled)
    Y_pred_scaled = np.array(Y_pred_scaled)
    y_pred = scaler.inverse_transform(Y_pred_scaled).flatten()

    return calculate_metrics(test_data['Units Sold'].values, y_pred, 'LSTM'), model, scaler, y_pred


def run_arima(train_data, test_data, order=(5, 1, 0)):
    """Runs ARIMA model for total store sales."""
    ts_train = train_data.set_index('Date')['Units Sold']
    ts_train.index.freq = 'D'
    model = ARIMA(ts_train, order=order)
    model_fit = model.fit()

    start = len(ts_train)
    end = len(ts_train) + len(test_data) - 1
    forecast = model_fit.predict(start=start, end=end, dynamic=False)
    y_pred = forecast.values
    return calculate_metrics(test_data['Units Sold'].values, y_pred, 'ARIMA'), model_fit, y_pred


def create_radar_chart(df_metrics):
    """Generates a radar chart comparing normalized model errors."""
    categories = [col for col in df_metrics.columns if col != 'Model']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    df_scaled = df_metrics.copy()
    for col in categories:
        min_val, max_val = df_metrics[col].min(), df_metrics[col].max()
        df_scaled[col] = (df_metrics[col] - min_val) / (max_val - min_val) if max_val != min_val else 0

    for i, row in df_scaled.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        label = df_metrics.loc[i, 'Model']
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(['Best', '25%', '50%', '75%', 'Worst'], color="grey")
    ax.set_ylim(0, 1)
    plt.title('Model Performance Comparison (Normalized Error)', size=16)
    plt.legend(loc='lower left', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
    plt.savefig('model_comparison_radar_chart_all.png')


# ======================= Main Script =======================

if __name__ == "__main__":
    FILE_PATH = "cleaned_sales_data.csv"
    SPLIT_DATE = '2023-12-01'
    df_raw = pd.read_csv(FILE_PATH)

    print("--- 1. Overall Store-Level Forecasting ---")
    train_data, test_data, y_test_true = prepare_data(df_raw, SPLIT_DATE)

    lstm_results, lstm_model, lstm_scaler, lstm_forecast = run_lstm(train_data, test_data)
    prophet_results, prophet_model, prophet_forecast = run_prophet(train_data, test_data)
    arima_results, arima_model, arima_forecast = run_arima(train_data, test_data)

    df_metrics = pd.DataFrame([lstm_results, prophet_results, arima_results])
    print("\n--- Overall Model Comparison ---")
    print(df_metrics.set_index('Model').to_markdown(floatfmt=".2f"))
    create_radar_chart(df_metrics)

    best_model = df_metrics.loc[df_metrics['MAPE'].idxmin(), 'Model']
    print(f"\n🏆 Best Performing Overall Model: {best_model}")

    if best_model == 'Prophet':
        model_to_save = {'model': prophet_model}
    elif best_model == 'LSTM':
        model_to_save = {'model': lstm_model, 'scaler': lstm_scaler}
    else:
        model_to_save = {'model': arima_model}
    with open(f'best_forecasting_model_{best_model.lower()}.pkl', 'wb') as f:
        pickle.dump(model_to_save, f)

    print("\n✅ Overall model comparison complete. Now forecasting per product...")

    # ======================= Per-Product Prophet Forecasting =======================
    all_forecasts, metrics_list = [], []

    for product_id, df_product in df_raw.groupby('Product ID'):
        try:
            train_data, test_data, _ = prepare_data(df_product, SPLIT_DATE)
            metrics, model, forecast = run_prophet(train_data, test_data)
            df_result = test_data.copy()
            df_result['Product ID'] = product_id
            df_result['Predicted Units Sold'] = forecast.round(0).astype(int)
            df_result['Absolute Error'] = np.abs(df_result['Units Sold'] - df_result['Predicted Units Sold'])
            all_forecasts.append(df_result)

            metrics['Product ID'] = product_id
            metrics_list.append(metrics)
        except Exception as e:
            print(f"⚠️ Skipped Product {product_id} due to error: {e}")

    df_all_forecasts = pd.concat(all_forecasts)
    df_all_forecasts.to_csv("forecast_per_product_prophet.csv", index=False)

    df_product_metrics = pd.DataFrame(metrics_list)
    df_product_metrics.to_csv("metrics_per_product_prophet.csv", index=False)

    print("\n✅ Per-product Prophet forecasts saved to 'forecast_per_product_prophet.csv'")
    print("✅ Per-product metrics saved to 'metrics_per_product_prophet.csv'")
    print("✅ Milestone 2 successfully completed 🚀")
