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

def create_future_projection(model, scaler_data, sales_history, start_date, category, all_categories, initial_stock):
    scaler = scaler_data['scaler']
    feature_names = scaler_data['feature_names']
    continuous_cols = scaler_data['continuous_cols']
    
    history = sales_history.tail(N_LAGS).tolist()
    stock_levels = [initial_stock]
    dates = []
    daily_q90_list = []
    
    curr_date = start_date
    current_stock = initial_stock
    
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
        feat_scaled = feat_df.copy()
        feat_scaled[continuous_cols] = scaler.transform(feat_df[continuous_cols])
        feat_t = torch.tensor(feat_scaled.values, dtype=torch.float32)
        
        lags_t = feat_t[:, :num_lags]
        static_t = feat_t[:, num_lags:]
        
        with torch.no_grad():
            preds_log = model(lags_t, static_t).numpy()[0]
        
        preds = np.expm1(preds_log)
        daily_q50 = preds[1]
        daily_q90 = preds[2]
        
        current_stock -= daily_q50
        stock_levels.append(current_stock)
        daily_q90_list.append(daily_q90)
        
        history.append(daily_q50)
        dates.append(curr_date)
        curr_date += timedelta(days=1)
        
    return dates, stock_levels[1:], daily_q90_list

def plot_inventory_projection(hist_dates, hist_stock, dates, stock_levels, item_name, lead_time, deadline_date):
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    plt.figure(figsize=(8, 4))
    
    # Объединяем для отрисовки общей линии
    all_dates = list(hist_dates) + list(dates)
    all_stock = list(hist_stock) + list(stock_levels)
    
    # История
    plt.plot(hist_dates, hist_stock, color='#7f8c8d', lw=1.5, label='История', linestyle='--')
    # Прогноз
    plt.plot(dates, stock_levels, color='#2980b9', lw=2.5, label='Прогноз запаса')
    
    # Заливка прогноза (только положительные значения)
    plt.fill_between(dates, 0, [max(0, s) for s in stock_levels], color='#3498db', alpha=0.2)
    
    # Линия нуля
    plt.axhline(0, color='#c0392b', linestyle='-', lw=1)
    
    # Точка обнуления
    oos_date = None
    for d, s in zip(dates, stock_levels):
        if s <= 0:
            oos_date = d
            break
            
    if oos_date:
        plt.axvline(oos_date, color='#e74c3c', linestyle='--', alpha=0.7)
        plt.scatter([oos_date], [0], color='#e74c3c', zorder=5)
        plt.text(oos_date, max(all_stock)*0.05, ' Обнуление', color='#e74c3c', fontsize=9, fontweight='bold')
        
    # Дедлайн заказа
    if deadline_date:
        is_past = deadline_date < dates[0]
        color = '#d35400' if is_past else '#f39c12'
        label = f'Дедлайн: {deadline_date.strftime("%d.%m")}' + (' (ПРОСРОЧЕНО)' if is_past else '')
        
        plt.axvline(deadline_date, color=color, linestyle='-.', lw=2, label=label)
        if is_past:
            plt.gca().axvspan(deadline_date, dates[0], color=color, alpha=0.1)

    plt.title(f"Прогноз движения запасов: {item_name}", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Запас (шт.)")
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(loc='upper right', fontsize=8, frameon=True, shadow=True)
    
    # Настройка осей
    plt.ylim(min(all_stock) if min(all_stock) < 0 else 0, max(all_stock) * 1.15)
    plt.xticks(rotation=25, fontsize=8)
    plt.tight_layout()
    
    img_filename = f"proj_{item_name.replace(' ', '_')}.png"
    img_path = os.path.join(PLOTS_DIR, img_filename)
    plt.savefig(img_path, dpi=120)
    plt.close()
    return img_path

def main():
    print("\n--- Этап 4: Генерация Плана Закупок (Улучшенная визуализация) ---")
    font_name = register_cyrillic_font()
    
    scaler_data = joblib.load(SCALER_PATH)
    feature_names = scaler_data['feature_names']
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
        
        # Получаем историю для графика (последние 14 дней)
        hist_view_days = 14
        hist_sales = prod_sales.tail(hist_view_days)
        hist_dates = hist_sales.index
        
        # Восстанавливаем исторический остаток (примерно)
        hist_stock_vals = []
        temp_stock = current_stock
        # Идем назад от текущего остатка
        for s_val in reversed(hist_sales.values):
            hist_stock_vals.insert(0, temp_stock + s_val)
            temp_stock += s_val
        
        dates, stocks, q90_demands = create_future_projection(
            model, scaler_data, prod_sales, prediction_start, category, all_cats, current_stock
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
            status = 'НОРМА'
            
        target_inventory = sum(q90_demands[:lead_time + COVERAGE_DAYS])
        order_qty = max(0, int(np.ceil(target_inventory - current_stock)))
        
        if not oos_date: order_qty = 0

        recommendations.append({
            'Категория': category[:15],
            'Товар': name,
            'Остаток': int(current_stock),
            'Склад пуст': oos_date.strftime('%d.%m') if oos_date else '30+ дн',
            'Заказать до': deadline_date.strftime('%d.%m') if deadline_date else '-',
            'Заказ': f"{order_qty} шт",
            'Статус': status
        })
        
        img_path = plot_inventory_projection(hist_dates, hist_stock_vals, dates, stocks, name, lead_time, deadline_date)
        graphs.append({'name': name, 'cat': category, 'img': img_path, 'status': status})

    # ГЕНЕРАЦИЯ PDF
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    now = datetime.now()
    doc_filename = f"Report_{now.strftime('%d_%m_%Y_%H_%M_%S')}.pdf"
    doc_path = os.path.join(REPORTS_DIR, doc_filename)
    
    doc = SimpleDocTemplate(doc_path, pagesize=landscape(letter), topMargin=20, bottomMargin=20)
    elements = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CustomTitle', fontName=font_name, fontSize=18, spaceAfter=20, alignment=1, textColor=colors.HexColor('#2c3e50')))
    styles['Normal'].fontName = font_name

    elements.append(Paragraph(f"ОТЧЕТ ПО ПОПОЛНЕНИЮ ЗАПАСОВ (LSTM ПРОГНОЗ)", styles['CustomTitle']))
    elements.append(Paragraph(f"Дата формирования: {now.strftime('%d.%m.%Y %H:%M')} | Период прогноза: 30 дней", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    df_rec = pd.DataFrame(recommendations)
    table_data = [df_rec.columns.tolist()] + df_rec.values.tolist()
    
    # Фиксированные ширины колонок для ландшафтной ориентации
    col_widths = [1.2*inch, 2.2*inch, 0.8*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.2*inch]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    style_list = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,-1), font_name),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
    ]
    
    for idx in range(len(df_rec)):
        row_idx = idx + 1
        st = df_rec.iloc[idx]['Статус']
        if st == 'КРИТИЧНО':
            style_list.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#fdeaea')))
            style_list.append(('TEXTCOLOR', (6, row_idx), (6, row_idx), colors.red))
        elif st == 'ПЛАНОВАЯ':
            style_list.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#fff9e6')))
            
    t.setStyle(TableStyle(style_list))
    elements.append(t)
    elements.append(PageBreak())
    
    # Секция с графиками
    elements.append(Paragraph("ДЕТАЛИЗАЦИЯ ПО ТОВАРАМ", styles['CustomTitle']))
    
    img_table_data = []
    row = []
    for g in graphs:
        # Увеличиваем размер графиков для лучшей читаемости
        img = Image(g['img'], width=4.8*inch, height=2.4*inch)
        row.append(img)
        if len(row) == 2:
            img_table_data.append(row)
            row = []
    if row:
        img_table_data.append(row + [''])
        
    img_t = Table(img_table_data, colWidths=[5*inch, 5*inch])
    elements.append(img_t)

    doc.build(elements)
    print(f"Обновленный отчет сформирован: {doc_path}")

if __name__ == '__main__':
    main()