import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import torch
import torch.nn as nn
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

class LSTMForecaster(nn.Module):
    def __init__(self, num_static_features, num_quantiles=len(QUANTILES)):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
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

def register_cyrillic_font():
    font_paths = [
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
        ("/System/Library/Fonts/Helvetica.ttc", None) # Fallback
    ]
    for reg_path, bold_path in font_paths:
        if os.path.exists(reg_path):
            pdfmetrics.registerFont(TTFont('CyrillicFont', reg_path))
            if bold_path and os.path.exists(bold_path):
                pdfmetrics.registerFont(TTFont('CyrillicFont-Bold', bold_path))
            else:
                # Если жирного нет, используем обычный как замену
                pdfmetrics.registerFont(TTFont('CyrillicFont-Bold', reg_path))
            return 'CyrillicFont'
    return 'Helvetica'

def create_future_projection(model, scaler, sales_history, start_date, category, all_categories, initial_stock):
    history = sales_history.tail(N_LAGS).tolist()
    stock_levels = [initial_stock]
    dates = []
    daily_q90_list = []
    
    curr_date = start_date
    current_stock = initial_stock
    
    feature_names = scaler.feature_names_in_
    lag_cols = [c for c in feature_names if 'lag_' in c]
    num_lags = len(lag_cols)
    
    for day_offset in range(FORECAST_HORIZON):
        hist_log = np.log1p(history[-N_LAGS:])
        features = {}
        
        # Лаги от старого к новому
        for i in range(N_LAGS, 0, -1):
            features[f'lag_{i}'] = hist_log[-i]
            
        features['rolling_mean_7'] = np.mean(hist_log[-7:])
        features['rolling_std_7'] = np.std(hist_log[-7:])
        
        doy = curr_date.dayofyear
        month = curr_date.month
        features['day_sin'] = np.sin(2 * np.pi * doy / 365.25)
        features['day_cos'] = np.cos(2 * np.pi * doy / 365.25)
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        features['is_weekend'] = 1 if curr_date.dayofweek >= 5 else 0
        
        # Симуляция будущих праздников и акций
        features['is_holiday'] = 1 if (curr_date.month == 12 and curr_date.day >= 25) or (curr_date.month == 1 and curr_date.day <= 5) else 0
        
        # Допустим, мы знаем, что акции будут в начале каждого месяца
        features['is_promo'] = 1 if curr_date.day <= 5 else 0 
        
        # Для прогноза мы ВСЕГДА предполагаем, что товар БУДЕТ на полке, 
        # иначе модель предскажет ноль продаж, и мы не закажем товар.
        features['in_stock'] = 1 
        
        for cat in all_categories:
            features[f'cat_{cat}'] = 1 if cat == category else 0
            
        feat_df = pd.DataFrame([features])[feature_names]
        feat_scaled = scaler.transform(feat_df)
        feat_t = torch.tensor(feat_scaled, dtype=torch.float32)
        
        lags_t = feat_t[:, :num_lags]
        static_t = feat_t[:, num_lags:]
        
        with torch.no_grad():
            preds_log = model(lags_t, static_t).numpy()[0]
        
        preds = np.expm1(preds_log)
        daily_q50 = preds[1] / 7 
        daily_q90 = preds[2] / 7
        
        current_stock -= daily_q50
        stock_levels.append(current_stock)
        daily_q90_list.append(daily_q90)
        
        history.append(daily_q50)
        dates.append(curr_date)
        curr_date += timedelta(days=1)
        
    return dates, stock_levels[1:], daily_q90_list

def plot_inventory_projection(dates, stock_levels, item_name, lead_time, deadline_date):
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    plt.figure(figsize=(7, 3.5))
    stock_visual = [max(0, s) for s in stock_levels]
    
    plt.plot(dates, stock_levels, color='#2980b9', lw=2, label='Прогноз запаса')
    plt.fill_between(dates, 0, stock_visual, color='#3498db', alpha=0.2)
    plt.axhline(0, color='#c0392b', linestyle='-', lw=1.5)
    
    oos_date = None
    for d, s in zip(dates, stock_levels):
        if s <= 0:
            oos_date = d
            break
            
    if oos_date:
        plt.axvline(oos_date, color='#e74c3c', linestyle='--', label='Склад пуст')
        plt.text(oos_date, max(stock_levels)*0.8, ' 0 шт', color='#e74c3c', fontweight='bold')
        
    if deadline_date and deadline_date >= dates[0]:
        plt.axvline(deadline_date, color='#f39c12', linestyle='-.', lw=2, label=f'Дедлайн заказа\n(LT={lead_time}дн)')
        
    plt.title(f"Траектория (LSTM): {item_name}", fontsize=11, pad=10)
    plt.ylabel("Штук на полке")
    plt.xticks(rotation=30, fontsize=8)
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    img_filename = f"proj_{item_name.replace(' ', '_')}.png"
    img_path = os.path.join(PLOTS_DIR, img_filename)
    plt.savefig(img_path, dpi=150)
    plt.close()
    return img_path

def main():
    print("\n--- Этап 4: Генерация Плана Закупок (LSTM + Внешние факторы) ---")
    font_name = register_cyrillic_font()
    
    scaler = joblib.load(SCALER_PATH)
    
    feature_names = scaler.feature_names_in_
    num_lags = len([c for c in feature_names if 'lag_' in c])
    num_static = len(feature_names) - num_lags
    
    model = LSTMForecaster(num_static_features=num_static)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with sqlite3.connect(DB_NAME) as conn:
        products = pd.read_sql_query("SELECT * FROM products", conn)
        sales = pd.read_sql_query("SELECT * FROM sales_history", conn, parse_dates=['sale_date'])
        stock = pd.read_sql_query("SELECT * FROM warehouse_stock", conn)

    prediction_start = sales['sale_date'].max() + timedelta(days=1)
    all_cats = products['category'].unique().tolist()
    
    recommendations = []
    graphs = []

    for _, prod in products.iterrows():
        pid = prod['id']
        name = prod['name']
        lead_time = prod['lead_time']
        category = prod['category']
        
        prod_sales = sales[sales['product_id'] == pid].set_index('sale_date')['quantity_sold'].sort_index()
        current_stock = stock[stock['product_id'] == pid]['current_quantity'].iloc[0]
        
        dates, stocks, q90_demands = create_future_projection(
            model, scaler, prod_sales, prediction_start, category, all_cats, current_stock
        )
        
        oos_date = None
        for d, s in zip(dates, stocks):
            if s <= 0:
                oos_date = d
                break
                
        if oos_date:
            deadline_date = oos_date - timedelta(days=lead_time)
            status = 'КРИТИЧНО' if deadline_date <= prediction_start else 'ПЛАНОВАЯ'
        else:
            deadline_date = None
            status = 'НОРМА (Остатка хватит на 30+ дней)'
            
        target_inventory = sum(q90_demands[:lead_time + COVERAGE_DAYS])
        order_qty = max(0, int(np.ceil(target_inventory - current_stock)))
        
        if not oos_date: order_qty = 0

        recommendations.append({
            'Категория': category,
            'Товар': name,
            'Остаток': int(current_stock),
            'Дата обнуления': oos_date.strftime('%d.%m.%Y') if oos_date else 'Более 30 дн.',
            'Дедлайн заказа': deadline_date.strftime('%d.%m.%Y') if deadline_date else '-',
            'Заказать (шт)': order_qty,
            'Статус': status,
            'LT (дн)': lead_time
        })
        
        img_path = plot_inventory_projection(dates, stocks, name, lead_time, deadline_date)
        graphs.append({'name': name, 'cat': category, 'img': img_path, 'status': status})

    # ГЕНЕРАЦИЯ PDF
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    now = datetime.now()
    doc_filename = f"Report_{now.strftime('%d_%m_%Y_%H_%M_%S')}.pdf"
    doc_path = os.path.join(REPORTS_DIR, doc_filename)
    
    doc = SimpleDocTemplate(doc_path, pagesize=landscape(letter), rightMargin=30, leftMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontName=font_name, fontSize=16, spaceAfter=20, alignment=1))
    styles['Normal'].fontName = font_name

    elements.append(Paragraph(f"ПЛАН ЗАКУПОК (LSTM ПРОГНОЗ) НА {prediction_start.strftime('%d.%m.%Y')}", styles['CustomTitle']))
    
    df_rec = pd.DataFrame(recommendations).sort_values(by=['Категория', 'Дедлайн заказа']).reset_index(drop=True)
    table_data = [df_rec.columns.tolist()] + df_rec.values.tolist()
    
    t = Table(table_data, repeatRows=1)
    style_list = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,-1), font_name),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]
    
    # Теперь итерируемся по строкам гарантированно правильно
    for idx in range(len(df_rec)):
        row_idx = idx + 1 # +1 так как 0 - это заголовок
        row = df_rec.iloc[idx]
        
        if row['Статус'] == 'КРИТИЧНО':
            style_list.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#ffeaea')))
            style_list.append(('TEXTCOLOR', (4, row_idx), (4, row_idx), colors.red)) # Дедлайн красным
            style_list.append(('FONTNAME', (1, row_idx), (1, row_idx), f"{font_name}-Bold" if font_name != 'Helvetica' else 'Helvetica-Bold'))
        elif row['Статус'] == 'ПЛАНОВАЯ':
            style_list.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#fffaea')))
        elif 'НОРМА' in row['Статус']:
            style_list.append(('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.HexColor('#7f8c8d')))
            
    t.setStyle(TableStyle(style_list))
    elements.append(t)
    elements.append(PageBreak())
    
    elements.append(Paragraph("ДЕТАЛИЗАЦИЯ ИСТОЩЕНИЯ ЗАПАСОВ (ГРАФИКИ LSTM)", styles['CustomTitle']))
    
    img_table_data = []
    row = []
    for g in graphs:
        img = Image(g['img'], width=4.5*inch, height=2.25*inch)
        row.append(img)
        if len(row) == 2:
            img_table_data.append(row)
            row = []
    if row:
        img_table_data.append(row + [''])
        
    img_t = Table(img_table_data)
    elements.append(img_t)

    doc.build(elements)
    print(f"Ультимативный план закупок готов: {doc_path}")

if __name__ == '__main__':
    main()