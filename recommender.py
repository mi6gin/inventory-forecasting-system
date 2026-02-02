
import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta, datetime
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie

# --- Константы ---
DB_NAME = "inventory_forecast.db"
MODEL_PATH = "demand_model.pkl"
SCALER_PATH = "demand_scaler.pkl"
N_LAGS = 35 # Должно совпадать с model_trainer.py
SAFETY_STOCK_FACTOR = 0.20  # 20% страховой запас
OVERSTOCK_FACTOR = 1.8      # Порог для избыточного запаса
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie

def create_pdf_report(report_df, prediction_date, filename):
    """Создает PDF-отчет с таблицей и графиком."""
    pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/Arial.ttf'))
    
    doc = SimpleDocTemplate(filename, pagesize=landscape(letter), title=f"Отчет по закупкам {prediction_date.strftime('%Y-%m-%d')}")
    elements = []
    styles = getSampleStyleSheet()
    styles['h1'].fontName = 'Arial'
    styles['h2'].fontName = 'Arial'

    # Title is removed from content
    
    # Image
    try:
        elements.append(Image("model_performance_enhanced.png", width=6.0*inch, height=6.0*inch))
        elements.append(Spacer(1, 0.2*inch))
    except Exception as e:
        print(f"Could not find image: {e}")

    # Table
    report_df_reset = report_df
    data = [report_df_reset.columns.to_list()] + report_df_reset.values.tolist()
    table = Table(data)
    
    # Styling
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F81BD')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Arial'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#DCE6F1')),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Arial')
    ])
    table.setStyle(style)
    elements.append(table)

    doc.build(elements)

def load_data_for_prediction():
    """Загружает все необходимые данные из БД."""
    with sqlite3.connect(DB_NAME) as conn:
        products = pd.read_sql_query("SELECT * FROM products", conn)
        sales = pd.read_sql_query("SELECT * FROM sales_history", conn, parse_dates=['sale_date'])
        stock = pd.read_sql_query("SELECT * FROM warehouse_stock", conn)
    return products, sales, stock

def create_prediction_features(sales_history, prediction_date):
    """Создает вектор признаков для прогнозирования на основе последней истории продаж."""
    # sales_history должен быть pd.Series с индексом-датой, отсортированный
    features = {}
    
    for i in range(1, N_LAGS + 1):
        features[f'lag_{i}'] = sales_history.iloc[-i]
        
    features['rolling_mean_7'] = sales_history.tail(7).mean()
    features['rolling_mean_30'] = sales_history.tail(30).mean()
    
    features['day_of_week'] = prediction_date.dayofweek
    features['month'] = prediction_date.month
    features['year'] = prediction_date.year
    features['day_of_year'] = prediction_date.dayofyear
    
    return pd.DataFrame([features])

def main():
    """Основная функция для генерации рекомендаций по закупкам."""
    print("\n--- Этап 4: Формирование управленческих рекомендаций ---")
    
    # 1. Загрузка артефактов и данных
    print("4.1. Загрузка модели, скейлера и данных из БД...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"Ошибка: Не найдены файлы '{MODEL_PATH}' или '{SCALER_PATH}'.")
        print("Сначала запустите 'model_trainer.py' для обучения модели.")
        return
        
    products_df, sales_df, stock_df = load_data_for_prediction()
    
    recommendations = []
    # Прогноз делаем на следующий день после последней записи в истории
    prediction_date = sales_df['sale_date'].max() + timedelta(days=1)
    print(f"4.2. Расчет прогноза и рекомендаций на дату: {prediction_date}")
    
    # 2. Итерация по продуктам и генерация рекомендаций
    for _, product in products_df.iterrows():
        pid = product['id']
        lead_time = product['lead_time']
        
        # История продаж для данного продукта
        prod_sales = sales_df[sales_df['product_id'] == pid].set_index('sale_date')['quantity_sold'].sort_index()
        
        if len(prod_sales) < N_LAGS:
            print(f"  - Пропуск продукта {pid}: недостаточно истории для создания признаков.")
            continue
            
        # Создание признаков для прогноза
        features_df = create_prediction_features(prod_sales, prediction_date)
        # Получаем порядок колонок, на котором обучалась модель
        model_feature_names = scaler.get_feature_names_out()
        features_df = features_df[model_feature_names]

        # Масштабирование и прогноз
        features_scaled = scaler.transform(features_df)
        predicted_demand = model.predict(features_scaled)[0]
        predicted_demand = max(0, predicted_demand)
        
        # Текущий остаток
        current_stock = stock_df[stock_df['product_id'] == pid]['current_quantity'].iloc[0]
        
        # Логика принятия решений
        safety_stock = predicted_demand * SAFETY_STOCK_FACTOR
        required_stock = predicted_demand + safety_stock
        purchase_volume = required_stock - current_stock
        
        if purchase_volume > 0:
            status = "Заказать"
            purchase_volume = int(np.ceil(purchase_volume))
            comment = f"Прогноз спроса на {lead_time} дней: {int(predicted_demand)} шт."
        else:
            if current_stock > required_stock * OVERSTOCK_FACTOR:
                status = "Стоп-заказ"
                comment = f"Риск переизбытка. Запас ({current_stock}) значительно превышает потребность ({int(required_stock)})."
            else:
                status = "Запаса достаточно"
                comment = "Текущих запасов хватает на период доставки."
            purchase_volume = 0

        recommendations.append({
            'ID товара': pid,
            'Название товара': product['name'],
            'Прогноз спроса': int(predicted_demand),
            'Текущий остаток': int(current_stock),
            'Страховой запас': int(safety_stock),
            'Рекомендация': status,
            'К закупке (шт.)': purchase_volume,
            'Комментарий': comment
        })
        
    # 3. Формирование отчета
    report_df = pd.DataFrame(recommendations)
    
    report_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dynamic_report_file = f"Отчет_по_закупкам_{report_time}.pdf"

    create_pdf_report(report_df, prediction_date, dynamic_report_file)
        
    print(f"\nОтчет сохранен в файл: {dynamic_report_file}")

if __name__ == '__main__':
    main()
