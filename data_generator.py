import pandas as pd
import numpy as np
import sqlite3
from datetime import date, timedelta

# --- Параметры генерации ---
N_PRODUCTS = 15
START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)
DB_NAME = "inventory_forecast.db"

def create_database_tables(conn):
    """Создает таблицы в базе данных, если они не существуют."""
    cursor = conn.cursor()
    
    # Таблица: products (id, название, категория, цена, lead_time — время поставки)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        price REAL NOT NULL,
        lead_time INTEGER NOT NULL
    )
    """)
    
    # Таблица: sales_history (id, product_id, дата, количество проданного товара)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        sale_date DATE NOT NULL,
        quantity_sold INTEGER NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    """)

    # Таблица: warehouse_stock (id, product_id, текущий остаток)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS warehouse_stock (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER UNIQUE NOT NULL,
        current_quantity INTEGER NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    """)
    
    conn.commit()

def generate_all_data():
    """Генерирует данные сразу для всех таблиц с комплексной сезонностью."""
    
    products_data = []
    sales_data = []
    stock_data = []

    categories = ['Электроника', 'Одежда', 'Продукты', 'Бытовая химия', 'Книги']
    
    for i in range(1, N_PRODUCTS + 1):
        # 1. Генерируем данные для таблицы `products`
        product_info = {
            'id': i,
            'name': f'Товар_{chr(64+i)}', # Товар_A, Товар_B...
            'category': np.random.choice(categories),
            'price': round(np.random.uniform(50, 8000), 2),
            'lead_time': np.random.randint(7, 25) # Время доставки в днях
        }
        products_data.append(product_info)

        # 2. Генерируем данные для таблицы `sales_history`
        dates = pd.to_datetime(pd.date_range(START_DATE, END_DATE, freq='D'))
        n_days = len(dates)
        
        base_sales = np.random.randint(10, 40)
        # Тренд роста/падения
        trend_factor = 1 + np.linspace(np.random.uniform(-0.5, 1.5), 1, n_days)
        
        # Сезонные колебания (годовые, месячные, недельные)
        days_of_year = np.arange(n_days)
        year_seasonality = 1 + 0.3 * np.sin(days_of_year * (2 * np.pi / 365.25))
        month_seasonality = 1 + 0.1 * np.sin(days_of_year * (2 * np.pi / 30.44))
        week_seasonality = 1 + 0.15 * np.sin(days_of_year * (2 * np.pi / 7) + np.random.uniform(0, 2*np.pi)) # сдвиг фазы для разных товаров
        
        seasonality_factor = year_seasonality * month_seasonality * week_seasonality

        # Случайный шум
        noise = np.random.normal(1, 0.1, n_days)
        
        sales_volume = base_sales * trend_factor * seasonality_factor * noise
        sales_volume = np.maximum(0, sales_volume).astype(int)

        current_day_stock = np.zeros(n_days, dtype=int)
        initial_stock = base_sales * product_info['lead_time'] * 1.2 # Начальный запас
        current_stock_level = initial_stock
        
        for j in range(n_days):
            sales_data.append({
                'product_id': i,
                'sale_date': dates[j].date(),
                'quantity_sold': sales_volume[j]
            })
            # Упрощенный расчет остатка на конец дня
            current_stock_level = max(0, current_stock_level - sales_volume[j])
            current_day_stock[j] = current_stock_level
            # Упрощенная логика пополнения: если остаток падает, заказываем (без моделирования lead_time)
            if current_stock_level < base_sales * 5:
                current_stock_level += initial_stock * np.random.uniform(0.5, 1.0)


        # 3. Сохраняем финальный остаток для `warehouse_stock`
        stock_data.append({
            'product_id': i,
            'current_quantity': int(current_day_stock[-1])
        })
        
    return pd.DataFrame(products_data), pd.DataFrame(sales_data), pd.DataFrame(stock_data)

def main():
    """Основная функция для генерации и сохранения данных в новую структуру БД."""
    print(f"1. Инициализация базы данных '{DB_NAME}'...")
    
    with sqlite3.connect(DB_NAME) as conn:
        create_database_tables(conn)
        
        print("2. Генерация синтетических данных (тренды, сезонность, шум)...")
        products_df, sales_df, stock_df = generate_all_data()
        
        print("3. Запись данных в таблицы...")
        products_df.to_sql('products', conn, if_exists='replace', index=False)
        sales_df.to_sql('sales_history', conn, if_exists='replace', index=False)
        stock_df.to_sql('warehouse_stock', conn, if_exists='replace', index=False)

    print("\nГенерация данных успешно завершена.")
    print(f"  - Записано {len(products_df)} продуктов.")
    print(f"  - Записано {len(sales_df)} записей о продажах.")
    print(f"  - Записано {len(stock_df)} записей о текущих остатках.")

if __name__ == '__main__':
    main()
