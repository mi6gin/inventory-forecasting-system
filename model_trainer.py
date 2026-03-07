import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import os
from losses import MultiQuantileLoss

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

DB_NAME = "inventory_forecast.db"
MODEL_PATH = "demand_model.pth"
SCALER_PATH = "demand_scaler.pkl"
N_LAGS = 35 
QUANTILES = [0.1, 0.5, 0.9]

class LSTMForecaster(nn.Module):
    def __init__(self, num_static_features, num_quantiles=len(QUANTILES)):
        super(LSTMForecaster, self).__init__()
        # Вход для LSTM: (batch, seq_len, 1) - только история продаж
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        
        # Полносвязная сеть объединяет выход LSTM и остальные (экзогенные) признаки
        self.fc = nn.Sequential(
            nn.Linear(64 + num_static_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_quantiles)
        )

    def forward(self, lags, static_features):
        lstm_out, _ = self.lstm(lags.unsqueeze(-1))
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat((last_hidden, static_features), dim=1)
        return torch.relu(self.fc(combined))

def add_temporal_features(df):
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    return df

def load_and_prepare_data(db_name):
    with sqlite3.connect(db_name) as conn:
        sales_df = pd.read_sql_query("SELECT product_id, sale_date, quantity_sold, in_stock, is_holiday, is_promo FROM sales_history", conn, parse_dates=['sale_date'])
        products_df = pd.read_sql_query("SELECT id as product_id, category, lead_time FROM products", conn)
    
    df = pd.merge(sales_df, products_df, on='product_id')
    df = df.set_index('sale_date').sort_index()
    
    cat_dummies = pd.get_dummies(df['category'], prefix='cat')
    df = pd.concat([df, cat_dummies], axis=1)
    return df

def create_features_and_target(df):
    df['quantity_sold_log'] = np.log1p(df['quantity_sold'])
    
    for i in range(1, N_LAGS + 1):
        df[f'lag_{i}'] = df.groupby('product_id')['quantity_sold_log'].shift(i)
    
    df['rolling_mean_7'] = df.groupby('product_id')['quantity_sold_log'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df.groupby('product_id')['quantity_sold_log'].shift(1).rolling(window=7).std()
    
    df = add_temporal_features(df)
    
    target_list = []
    for pid, group in df.groupby('product_id'):
        lt = group['lead_time'].iloc[0]
        target_sum = group['quantity_sold'].shift(-lt).rolling(window=lt).sum()
        group['target'] = np.log1p(target_sum)
        target_list.append(group)
    
    featured_df = pd.concat(target_list).dropna()
    return featured_df

def main():
    print("--- Этап 2: Обучение LSTM модели (С нуля) ---")
    df = load_and_prepare_data(DB_NAME)
    featured_df = create_features_and_target(df)
    
    # ЛАГИ: от старого к новому для временного ряда LSTM
    lag_cols = [f'lag_{i}' for i in range(N_LAGS, 0, -1)]
    # ОСТАЛЬНОЕ: экзогенные факторы
    static_cols = [c for c in featured_df.columns if 'rolling_' in c or 'sin' in c or 'cos' in c or 'is_' in c or 'cat_' in c or c == 'in_stock']
    
    feature_cols = lag_cols + static_cols
    X, y = featured_df[feature_cols], featured_df['target']
    
    split_date = featured_df.index.max() - pd.DateOffset(days=30)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X[X.index <= split_date])
    X_test_scaled = scaler.transform(X[X.index > split_date])
    joblib.dump(scaler, SCALER_PATH)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y[y.index <= split_date].values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y[y.index > split_date].values, dtype=torch.float32).unsqueeze(1)
    
    num_lags = len(lag_cols)
    num_static = len(static_cols)
    
    model = LSTMForecaster(num_static_features=num_static)
    criterion = MultiQuantileLoss(quantiles=QUANTILES)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True
    )

    epochs = 40
    print("Начинаем обучение LSTM...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for b_x, b_y in train_loader:
            lags = b_x[:, :num_lags]
            static = b_x[:, num_lags:]
            
            optimizer.zero_grad()
            loss = criterion(model(lags, static), b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 10 == 0: 
            print(f"  Эпоха {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), MODEL_PATH)

    print("\n--- Этап 3: Анализ качества ---")
    model.eval()
    with torch.no_grad():
        lags_test = X_test_t[:, :num_lags]
        static_test = X_test_t[:, num_lags:]
        preds_log = model(lags_test, static_test).numpy()
        preds_real = np.expm1(preds_log)
        y_test_real = np.expm1(y_test_t.numpy())
    
    r2_val = r2_score(y_test_real, preds_real[:, 1])
    mae_val = mean_absolute_error(y_test_real, preds_real[:, 1])
    print(f"  - R-squared (медиана): {r2_val:.4f}")
    print(f"  - MAE (средняя ошибка в штуках): {mae_val:.2f}")

if __name__ == '__main__':
    main()