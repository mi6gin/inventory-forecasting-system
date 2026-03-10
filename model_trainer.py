import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import warnings
import os
import math
from losses import MultiQuantileLoss
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

DB_NAME = "inventory_forecast.db"
MODEL_PATH = "demand_model.pth"
SCALER_PATH = "demand_scaler.pkl"
N_LAGS = 35 
QUANTILES = [0.1, 0.5, 0.9]

def wape_score(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-9) * 100

def recover_oos_demand(df):
    df = df.copy()
    df['recovered_demand'] = pd.to_numeric(df['quantity_sold'], errors='coerce').fillna(0).astype(float)
    for pid, group in df.groupby('product_id'):
        oos_mask = (group['in_stock'] == 0)
        if oos_mask.any():
            valid_demand = group.loc[~oos_mask, 'recovered_demand']
            if not valid_demand.empty:
                rolling_avg = valid_demand.rolling(window=7, min_periods=1).mean()
                full_avg = rolling_avg.reindex(group.index).ffill().fillna(valid_demand.mean())
                df.loc[group.index[oos_mask], 'recovered_demand'] = full_avg[oos_mask]
    return df

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class LSTMForecaster(nn.Module):
    def __init__(self, num_static_features, num_quantiles=len(QUANTILES)):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64 + num_static_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_quantiles))
    def forward(self, lags, static_features):
        lstm_out, _ = self.lstm(lags.unsqueeze(-1))
        combined = torch.cat((lstm_out[:, -1, :], static_features), dim=1)
        return torch.relu(self.fc(combined))

class GRUForecaster(nn.Module):
    def __init__(self, num_static_features, num_quantiles=len(QUANTILES)):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64 + num_static_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_quantiles))
    def forward(self, lags, static_features):
        gru_out, _ = self.gru(lags.unsqueeze(-1))
        combined = torch.cat((gru_out[:, -1, :], static_features), dim=1)
        return torch.relu(self.fc(combined))

class TransformerForecaster(nn.Module):
    def __init__(self, num_static_features, num_quantiles=len(QUANTILES)):
        super(TransformerForecaster, self).__init__()
        self.d_model = 64
        self.feature_proj = nn.Linear(1, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(self.d_model + num_static_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_quantiles))
    def forward(self, lags, static_features):
        src = self.pos_encoder(self.feature_proj(lags.unsqueeze(-1)))
        output = self.transformer_encoder(src)
        combined = torch.cat((output[:, -1, :], static_features), dim=1)
        return torch.relu(self.fc(combined))

def load_and_prepare_data(db_name):
    with sqlite3.connect(db_name) as conn:
        sales_df = pd.read_sql_query("SELECT * FROM sales_history", conn, parse_dates=['sale_date'])
        products_df = pd.read_sql_query("SELECT * FROM products", conn)
    sales_df['quantity_sold'] = pd.to_numeric(sales_df['quantity_sold'], errors='coerce').fillna(0)
    sales_df['in_stock'] = pd.to_numeric(sales_df['in_stock'], errors='coerce').fillna(1)
    df = pd.merge(sales_df, products_df, left_on='product_id', right_on='id')
    df = df.set_index('sale_date').sort_index()
    df = recover_oos_demand(df)
    cat_dummies = pd.get_dummies(df['category'], prefix='cat')
    df = pd.concat([df, cat_dummies], axis=1)
    return df

def create_features_and_target(df):
    df['quantity_recovered_log'] = np.log1p(df['recovered_demand'])
    for i in range(1, N_LAGS + 1):
        df[f'lag_{i}'] = df.groupby('product_id')['quantity_recovered_log'].shift(i)
    df['rolling_mean_7'] = df.groupby('product_id')['quantity_recovered_log'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df.groupby('product_id')['quantity_recovered_log'].shift(1).rolling(window=7).std()
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['target'] = df['quantity_recovered_log']
    return df.dropna()

def train_neural_model(model_class, X_data, y_data, num_lags, num_static, continuous_cols, feature_cols, epochs=30):
    tscv = TimeSeriesSplit(n_splits=3)
    fold_wapes = []
    last_model, last_scaler = None, None
    for _, (train_idx, test_idx) in enumerate(tscv.split(X_data)):
        X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
        scaler = RobustScaler(); scaler.fit(X_train[continuous_cols])
        def scale(X_df, sc):
            X_s = X_df.copy().values.astype(np.float32)
            cont_idx = [feature_cols.index(c) for c in continuous_cols]
            X_s_cont = sc.transform(X_df[continuous_cols])
            for i, idx in enumerate(cont_idx): X_s[:, idx] = X_s_cont[:, i]
            return X_s
        X_train_s, X_test_s = scale(X_train, scaler), scale(X_test, scaler)
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        model = model_class(num_static_features=num_static)
        criterion = MultiQuantileLoss(quantiles=QUANTILES)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)
        for _ in range(epochs):
            model.train()
            for b_x, b_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(b_x[:, :num_lags], b_x[:, num_lags:]), b_y)
                loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            preds_log = model(X_test_t[:, :num_lags], X_test_t[:, num_lags:]).numpy()[:, 1]
            fold_wapes.append(wape_score(np.expm1(y_test.values), np.expm1(preds_log)))
        last_model, last_scaler = model, scaler
    return last_model, np.mean(fold_wapes), last_scaler

def main():
    print("--- Этап 2: Обучение и честное сравнение (Metric: WAPE) ---")
    df = load_and_prepare_data(DB_NAME)
    featured_df = create_features_and_target(df)
    lag_cols = [f'lag_{i}' for i in range(N_LAGS, 0, -1)]
    static_cols = [c for c in featured_df.columns if 'rolling_' in c or 'sin' in c or 'cos' in c or 'is_' in c or 'cat_' in c or c == 'in_stock']
    feature_cols = lag_cols + static_cols
    continuous_cols = lag_cols + ['rolling_mean_7', 'rolling_std_7']
    X, y = featured_df[feature_cols], featured_df['target']
    num_lags, num_static = len(lag_cols), len(static_cols)
    tscv = TimeSeriesSplit(n_splits=3)
    _, test_idx = list(tscv.split(X))[-1]
    test_cutoff_date = X.index[test_idx[0]]
    results = {}
    best_wape, best_name, best_obj, best_sc = float('inf'), "", None, None
    for name, m_class in {"LSTM": LSTMForecaster, "GRU": GRUForecaster, "Transformer": TransformerForecaster}.items():
        print(f"Оценка {name}...")
        m_obj, avg_wape, sc = train_neural_model(m_class, X, y, num_lags, num_static, continuous_cols, feature_cols)
        results[name] = avg_wape
        if avg_wape < best_wape: best_wape, best_name, best_obj, best_sc = avg_wape, name, m_obj, sc
    print("Оценка ARIMA...")
    arima_wapes = []
    pids = featured_df['product_id'].unique()
    sample_pids = np.random.choice(pids, size=min(10, len(pids)), replace=False)
    for pid in sample_pids:
        p_df = featured_df[featured_df['product_id'] == pid]
        train_d = np.expm1(p_df.loc[p_df.index < test_cutoff_date, 'target'].values)
        test_d = np.expm1(p_df.loc[p_df.index >= test_cutoff_date, 'target'].values)
        if len(train_d) > 20 and len(test_d) > 0:
            try:
                m_arima = ARIMA(train_d, order=(5,1,0)).fit()
                arima_wapes.append(wape_score(test_d, m_arima.forecast(steps=len(test_d))))
            except: continue
    results["ARIMA"] = np.mean(arima_wapes) if arima_wapes else 100.0
    print(f"\n--- Победитель: {best_name} (WAPE: {best_wape:.2f}%) ---")
    torch.save(best_obj.state_dict(), MODEL_PATH)
    joblib.dump({'scaler': best_sc, 'feature_names': feature_cols, 'continuous_cols': continuous_cols, 'best_model_type': best_name, 'best_wape': best_wape}, SCALER_PATH)
    print("\n" + "="*40 + "\n" + f"{'Модель':<15} | {'Avg WAPE (%)':<10}\n" + "-"*40)
    for k, v in results.items(): print(f"{k:<15} | {v:<10.2f}")
    print("="*40)

if __name__ == '__main__': main()