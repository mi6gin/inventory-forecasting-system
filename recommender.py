import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import torch
import torch.nn as nn
import math
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

DB_NAME = "inventory_forecast.db"
MODEL_PATH = "demand_model.pth"
SCALER_PATH = "demand_scaler.pkl"
PLOTS_DIR = "charts"
REPORTS_DIR = "reports"
N_LAGS = 35 
QUANTILES = [0.1, 0.5, 0.9]
FORECAST_HORIZON = 30
COVERAGE_DAYS = 14

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

def register_font():
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(p):
            pdfmetrics.registerFont(TTFont('Cyrillic', p))
            return 'Cyrillic'
    return 'Helvetica'

def create_projection(model, scaler_data, sales_history, start_date, category, all_categories, initial_stock):
    scaler, feat_names, cont_cols = scaler_data['scaler'], scaler_data['feature_names'], scaler_data['continuous_cols']
    history = sales_history.tail(N_LAGS).tolist()
    stock_levels = [initial_stock]
    dates, q50_list, q90_list = [], [], []
    curr_date, current_stock = start_date, initial_stock
    num_lags = len([c for c in feat_names if 'lag_' in c])
    for _ in range(FORECAST_HORIZON):
        h_log = np.log1p(history[-N_LAGS:])
        feats = {f'lag_{i}': h_log[-i] for i in range(N_LAGS, 0, -1)}
        feats['rolling_mean_7'] = np.mean(h_log[-7:]); feats['rolling_std_7'] = np.std(h_log[-7:])
        feats['day_sin'] = np.sin(2 * np.pi * curr_date.dayofyear / 365.25); feats['day_cos'] = np.cos(2 * np.pi * curr_date.dayofyear / 365.25)
        feats['is_weekend'] = 1 if curr_date.dayofweek >= 5 else 0; feats['in_stock'] = 1 
        for cat in all_categories: feats[f'cat_{cat}'] = 1 if cat == category else 0
        f_df = pd.DataFrame([feats])
        for col in feat_names: 
            if col not in f_df.columns: f_df[col] = 0
        f_df = f_df[feat_names]
        f_s = f_df.copy(); f_s[cont_cols] = scaler.transform(f_df[cont_cols])
        f_t = torch.tensor(f_s.values, dtype=torch.float32)
        with torch.no_grad():
            preds = np.expm1(model(f_t[:, :num_lags], f_t[:, num_lags:]).numpy()[0])
        q50, q90 = preds[1], preds[2]
        current_stock -= q50; stock_levels.append(current_stock)
        q50_list.append(q50); q90_list.append(q90); history.append(q50); dates.append(curr_date); curr_date += timedelta(days=1)
    return dates, stock_levels[1:], q50_list, q90_list

def main():
    print("\n--- Этап 4: Генерация Плана Закупок (Lead Time Demand & In-Transit) ---")
    f_name = register_font()
    if not os.path.exists(SCALER_PATH): return
    s_data = joblib.load(SCALER_PATH)
    m_type, m_classes = s_data['best_model_type'], {"LSTM": LSTMForecaster, "GRU": GRUForecaster, "Transformer": TransformerForecaster}
    num_lags = len([c for c in s_data['feature_names'] if 'lag_' in c])
    model = m_classes[m_type](len(s_data['feature_names']) - num_lags)
    model.load_state_dict(torch.load(MODEL_PATH)); model.eval()
    with sqlite3.connect(DB_NAME) as conn:
        products = pd.read_sql_query("SELECT * FROM products", conn)
        sales = pd.read_sql_query("SELECT * FROM sales_history", conn, parse_dates=['sale_date'])
        stock = pd.read_sql_query("SELECT * FROM warehouse_stock", conn)
        try:
            transit = pd.read_sql_query("SELECT * FROM warehouse_in_transit", conn)
        except:
            transit = pd.DataFrame({'product_id': products['id'], 'in_transit_quantity': 0})
    sales['quantity_sold'] = pd.to_numeric(sales['quantity_sold'], errors='coerce').fillna(0)
    stock['current_quantity'] = pd.to_numeric(stock['current_quantity'], errors='coerce').fillna(0)
    transit['in_transit_quantity'] = pd.to_numeric(transit['in_transit_quantity'], errors='coerce').fillna(0)
    pred_start, all_cats, recs = sales['sale_date'].max() + timedelta(days=1), products['category'].unique().tolist(), []
    for _, prod in products.iterrows():
        pid, name, lt, cat = prod['id'], prod['name'], prod['lead_time'], prod['category']
        curr_stock = float(stock[stock['product_id'] == pid]['current_quantity'].iloc[0])
        in_transit = float(transit[transit['product_id'] == pid]['in_transit_quantity'].iloc[0])
        dates, stocks, q50s, q90s = create_projection(model, s_data, sales[sales['product_id'] == pid].set_index('sale_date')['quantity_sold'], pred_start, cat, all_cats, curr_stock)
        oos_date = next((d for d, s in zip(dates, stocks) if s <= 0), None)
        deadline = oos_date - timedelta(days=lt) if oos_date else None
        window = lt + COVERAGE_DAYS
        ltd_q50 = sum(q50s[:window])
        sigmas = [(q90 - q50)/1.28 for q50, q90 in zip(q50s[:window], q90s[:window])]
        safety_stock = 1.28 * np.sqrt(sum([s**2 for s in sigmas]))
        target_inv = ltd_q50 + safety_stock
        order_qty = max(0, int(np.ceil(target_inv - curr_stock - in_transit))) if oos_date else 0
        recs.append({'Категория': cat[:10], 'Товар': name, 'Остаток': int(curr_stock), 'В пути': int(in_transit), 'Обнуление': oos_date.strftime('%d.%m') if oos_date else '30+ дн', 'Дедлайн': deadline.strftime('%d.%m') if deadline else '-', 'Заказ': f"{order_qty} шт", 'WAPE': f"{s_data['best_wape']:.1f}%", 'Статус': 'КРИТИЧНО' if deadline and deadline <= pred_start else 'НОРМА'})
    if not os.path.exists(REPORTS_DIR): os.makedirs(REPORTS_DIR)
    doc_path = os.path.join(REPORTS_DIR, f"Scientific_Report_{datetime.now().strftime('%d_%m_%H_%M')}.pdf")
    doc = SimpleDocTemplate(doc_path, pagesize=landscape(letter))
    elements = [Paragraph(f"НАУЧНО-ОБОСНОВАННЫЙ ПЛАН ЗАКУПОК (Model: {m_type})", getSampleStyleSheet()['Heading1']), Spacer(1, 12)]
    df_recs = pd.DataFrame(recs); t = Table([df_recs.columns.tolist()] + df_recs.values.tolist())
    t.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), f_name), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTSIZE', (0,0), (-1,-1), 8)]))
    elements.append(t); doc.build(elements)
    print(f"Научный отчет сформирован: {doc_path}")

if __name__ == '__main__': main()