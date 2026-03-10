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
    print("--- Этап 1: Генерация данных (v3: In-Transit & OOS) ---")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    for table in ["sales_history", "warehouse_stock", "products", "warehouse_in_transit"]:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    
    cursor.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, lead_time INTEGER, unit_price REAL)")
    cursor.execute("CREATE TABLE sales_history (product_id INTEGER, sale_date DATE, quantity_sold INTEGER, in_stock INTEGER, is_holiday INTEGER, is_promo INTEGER)")
    cursor.execute("CREATE TABLE warehouse_stock (product_id INTEGER, current_quantity INTEGER)")
    cursor.execute("CREATE TABLE warehouse_in_transit (product_id INTEGER, in_transit_quantity INTEGER)")

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
            oos_days = 0
            records = []

            for d in range(days):
                curr_date = start_date + timedelta(days=d)
                is_holiday = 1 if (curr_date.month == 12 and curr_date.day >= 25) or (curr_date.month == 1 and curr_date.day <= 5) else 0
                is_promo = 1 if (d % 30) < 5 else 0 
                
                if oos_days > 0:
                    in_stock, oos_days = 0, oos_days - 1
                else:
                    in_stock = 0 if np.random.rand() < 0.02 else 1
                    if in_stock == 0: oos_days = np.random.randint(1, 4)
                
                demand = base_demand * (1.5 if is_holiday else 1.0) * (1.3 if curr_date.weekday() >= 5 else 1.0)
                quantity_sold = np.random.poisson(demand) if in_stock else 0
                records.append((product_id, curr_date.date(), quantity_sold, in_stock, is_holiday, is_promo))
            
            cursor.executemany("INSERT INTO sales_history VALUES (?,?,?,?,?,?)", records)
            cursor.execute("INSERT INTO warehouse_stock VALUES (?,?)", (product_id, np.random.randint(50, 300)))
            # Генерируем случайный товар в пути для некоторых позиций
            in_transit = np.random.choice([0, 100, 200], p=[0.7, 0.2, 0.1])
            cursor.execute("INSERT INTO warehouse_in_transit VALUES (?,?)", (product_id, in_transit))
            product_id += 1

    conn.commit()
    conn.close()
    print("Данные успешно обновлены (добавлена таблица In-Transit).")

if __name__ == "__main__":
    main()