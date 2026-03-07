import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DB_NAME = "inventory_forecast.db"

CATEGORIES = {
    "Овощи и фрукты": ["Яблоки", "Томаты", "Бананы", "Огурцы"],
    "Холодные напитки": ["Кола", "Вода 0.5л", "Сок яблочный", "Энергетик"],
    "Молочные продукты": ["Молоко 1л", "Йогурт", "Сыр Гауда", "Творог"],
    "Бакалея": ["Рис", "Гречка", "Макароны", "Сахар"]
}

def main():
    print("--- Этап 1: Генерация данных (включая Праздники, Акции и Дефицит) ---")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS sales_history")
    cursor.execute("DROP TABLE IF EXISTS warehouse_stock")
    cursor.execute("DROP TABLE IF EXISTS products")
    cursor.execute("DROP TABLE IF EXISTS forecast_history") # Убираем старую таблицу "памяти"
    
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            lead_time INTEGER,
            unit_price REAL
        )
    """)
    
    # Добавляем новые фичи: in_stock (наличие на полке), is_holiday (праздник), is_promo (акция)
    cursor.execute("CREATE TABLE sales_history (product_id INTEGER, sale_date DATE, quantity_sold INTEGER, in_stock INTEGER, is_holiday INTEGER, is_promo INTEGER)")
    cursor.execute("CREATE TABLE warehouse_stock (product_id INTEGER, current_quantity INTEGER)")

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 3, 1)
    days = (end_date - start_date).days

    product_id = 1
    for cat, items in CATEGORIES.items():
        for item in items:
            lt = 2 if cat == "Овощи и фрукты" else (7 if cat == "Молочные продукты" else 14)
            price = np.random.uniform(50, 500)
            cursor.execute("INSERT INTO products VALUES (?,?,?,?,?)", (product_id, item, cat, lt, price))
            
            base_demand = np.random.randint(20, 100)
            oos_days_remaining = 0

            for d in range(days):
                curr_date = start_date + timedelta(days=d)
                
                # Праздники: Новый год (25 дек - 5 янв)
                is_holiday = 1 if (curr_date.month == 12 and curr_date.day >= 25) or (curr_date.month == 1 and curr_date.day <= 5) else 0
                
                # Акции: 5 дней каждые 30 дней
                is_promo = 1 if (d % 30) < 5 else 0 
                
                # Логика дефицита (Out of Stock)
                if oos_days_remaining > 0:
                    in_stock = 0
                    oos_days_remaining -= 1
                else:
                    in_stock = 1
                    # 2% шанс, что товар внезапно закончится на 1-3 дня
                    if np.random.rand() < 0.02: 
                        oos_days_remaining = np.random.randint(1, 4)
                        in_stock = 0
                
                season_factor = 1.5 if (cat == "Холодные напитки" and curr_date.month in [6,7,8]) else 1.0
                day_factor = 1.3 if curr_date.weekday() >= 5 else 1.0
                holiday_factor = 1.4 if is_holiday else 1.0
                promo_factor = 1.5 if is_promo else 1.0
                
                if in_stock == 0:
                    quantity_sold = 0 # Продаж нет, если товара нет (дефицит)
                else:
                    demand = base_demand * season_factor * day_factor * holiday_factor * promo_factor
                    quantity_sold = np.random.poisson(demand)
                    
                cursor.execute("INSERT INTO sales_history VALUES (?,?,?,?,?,?)", 
                               (product_id, curr_date.date(), quantity_sold, in_stock, is_holiday, is_promo))
            
            cursor.execute("INSERT INTO warehouse_stock VALUES (?,?)", (product_id, np.random.randint(100, 500)))
            product_id += 1

    conn.commit()
    conn.close()
    print("Данные сгенерированы (добавлены фичи: in_stock, is_holiday, is_promo).")

if __name__ == "__main__":
    main()