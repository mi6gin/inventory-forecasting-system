import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Константы ---
DB_NAME = "inventory_forecast.db"
MODEL_PATH = "demand_model.pth"
SCALER_PATH = "demand_scaler.pkl"
N_LAGS = 35 # Увеличим глубину лагов для лучшего захвата зависимостей

def load_and_prepare_data(db_name):
    """Загружает и объединяет данные из SQLite."""
    with sqlite3.connect(db_name) as conn:
        sales_df = pd.read_sql_query("SELECT product_id, sale_date, quantity_sold FROM sales_history", conn, parse_dates=['sale_date'])
        products_df = pd.read_sql_query("SELECT id as product_id, lead_time FROM products", conn)
    
    # Объединяем данные
    df = pd.merge(sales_df, products_df, on='product_id')
    df = df.set_index('sale_date').sort_index()
    return df

def clean_data(df):
    """
    Выполняет очистку данных.
    (В этой реализации - заглушка, т.к. данные синтетические)
    """
    # Пример: удаление выбросов по методу межквартильного размаха
    # Q1 = df['quantity_sold'].quantile(0.25)
    # Q3 = df['quantity_sold'].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # df = df[(df['quantity_sold'] >= lower_bound) & (df['quantity_sold'] <= upper_bound)]
    
    # Заполнение пропусков (если они есть)
    df['quantity_sold'] = df['quantity_sold'].ffill()
    return df

def create_features_and_target(df):
    """Создает признаки (скользящее окно) и динамическую целевую переменную."""
    
    # Создание признаков
    for i in range(1, N_LAGS + 1):
        df[f'lag_{i}'] = df.groupby('product_id')['quantity_sold'].shift(i)
    
    df['rolling_mean_7'] = df.groupby('product_id')['quantity_sold'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_mean_30'] = df.groupby('product_id')['quantity_sold'].shift(1).rolling(window=30, min_periods=1).mean()
    
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear

    # Создание динамической целевой переменной
    # Горизонт прогноза = lead_time
    target_list = []
    for pid, group in df.groupby('product_id'):
        lead_time = group['lead_time'].iloc[0]
        # Сумма продаж за будущий период, равный lead_time
        group['target'] = group['quantity_sold'].shift(-lead_time).rolling(window=lead_time).sum()
        target_list.append(group)
    
    featured_df = pd.concat(target_list)
    featured_df = featured_df.dropna() # Удаляем строки, где признаки или цель не могут быть вычислены
    
    return featured_df

def mean_absolute_percentage_error(y_true, y_pred): 
    """Расчет MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Избегаем деления на ноль
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_enhanced_performance_chart(y_test, predictions, mae, rmse, mape):
    """
    Создает улучшенный и более понятный график оценки производительности модели.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))

    # --- Основной график ---
    ax.scatter(y_test, predictions, alpha=0.5, label="Точки прогнозов")
    
    # --- Линия идеального прогноза ---
    perfect_line_range = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(perfect_line_range, perfect_line_range, '--', color='red', lw=2, label="Идеальный прогноз")

    # --- Зоны ошибок ---
    ax.fill_between(perfect_line_range, perfect_line_range, perfect_line_range[1], color='orange', alpha=0.2, label="Зона переоценки (риск излишков)")
    ax.fill_between(perfect_line_range, perfect_line_range, perfect_line_range[0], color='blue', alpha=0.2, label="Зона недооценки (риск дефицита)")

    # --- Аннотации и текст ---
    ax.set_xlabel("Фактический спрос (сумма продаж за время поставки)", fontsize=12)
    ax.set_ylabel("Предсказанный спрос", fontsize=12)
    ax.set_title("Анализ точности модели: Где модель ошибается?", fontsize=16, fontweight='bold')
    
    # Текстовый блок с пояснениями
    explanation_text = (
        "КАК ЧИТАТЬ ГРАФИК:\n"
        "  - Каждая точка - это один прогноз для одного товара.\n"
        "  - Красная линия (---) - это идеальный прогноз, где предсказание = факт.\n"
        "  - Точки в ОРАНЖЕВОЙ зоне: модель предсказала больше, чем продали (риск излишков).\n"
        "  - Точки в СИНЕЙ зоне: модель предсказала меньше, чем продали (риск дефицита)."
    )
    ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Текстовый блок с метриками
    metrics_text = (
        f"Ключевые метрики:\n"
        f"  - MAE: {mae:.2f} (средняя ошибка в штуках)\n"
        f"  - MAPE: {mape:.2f}% (средняя ошибка в %)"
    )
    ax.text(0.65, 0.15, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.5))
    
    ax.legend(loc='lower right')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_performance_enhanced.png')
    plt.close(fig)

def main():
    """Основной пайплайн обучения модели."""
    print("--- Этап 2: Обучение модели ---")
    
    # 1. Загрузка и предобработка
    print("1.1. Загрузка и объединение данных...")
    df = load_and_prepare_data(DB_NAME)
    
    print("1.2. Очистка данных (пропуски, выбросы)...")
    df = clean_data(df)
    
    print("1.3. Создание признаков и динамической целевой переменной...")
    featured_df = create_features_and_target(df)
    
    # 2. Обучение модели
    print("2.1. Подготовка выборок для обучения...")
    features = [col for col in featured_df.columns if 'lag_' in col or 'rolling_' in col or 'day_' in col or 'month' in col or 'year' in col]
    X = featured_df[features]
    y = featured_df['target']
    
    # Разделение на обучающую и тестовую выборки (последние 9 месяцев - тест)
    split_date = featured_df.index.max() - pd.DateOffset(months=9)
    X_train, y_train = X[X.index <= split_date], y[y.index <= split_date]
    X_test, y_test = X[X.index > split_date], y[y.index > split_date]

    print(f"  - Размер обучающей выборки: {len(X_train)} записей")
    print(f"  - Размер тестовой выборки: {len(X_test)} записей")
    
    print("2.2. Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("2.3. Обучение нейронной сети PyTorch...")
    
    # Конвертация данных в тензоры PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # unsqueeze для правильной формы

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    # Инициализация модели, функции потерь и оптимизатора
    input_size = X_train_tensor.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение модели
    num_epochs = 100 # Количество эпох, можно настроить
    batch_size = 64 # Размер батча, можно настроить

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train() # Устанавливаем модель в режим обучения
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("2.4. Сохранение модели и скейлера...")
    # Сохраняем только состояние модели (state_dict)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # 3. Оценка качества
    print("\n--- Этап 3: Оценка качества модели ---")
    model.eval() # Устанавливаем модель в режим оценки
    with torch.no_grad():
        predictions_tensor = model(X_test_tensor)
        predictions = predictions_tensor.numpy().flatten()

    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    print(f"  - MAE (Mean Absolute Error): {mae:.2f}")
    print(f"  - RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"  - MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    print("3.1. Построение графика 'Прогноз vs. Факт'...")
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
    plt.xlabel('Фактические значения (сумма продаж за lead_time)')
    plt.ylabel('Предсказанные значения')
    plt.title('Оценка качества модели: Прогноз vs. Факт')
    plt.grid(True)
    plt.savefig('model_performance.png')
    print("  - График сохранен в 'model_performance.png'")

    print("3.2. Построение УЛУЧШЕННОГО графика 'Прогноз vs. Факт'...")
    create_enhanced_performance_chart(y_test, predictions, mae, rmse, mape)
    print("  - Улучшенный график сохранен в 'model_performance_enhanced.png'")

if __name__ == '__main__':
    main()
