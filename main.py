import os
import data_generator
import model_trainer
import recommender

def main():
    print("="*80)
    print("Запуск системы прогнозирования спроса (Multi-Model Comparison: LSTM, GRU, Transformer, ARIMA)")
    print("="*80)
    
    # --- Этап 1: Инфраструктура данных ---
    print("\n[ШАГ 1] Подготовка данных и восстановление OOS...")
    if not os.path.exists(data_generator.DB_NAME):
        data_generator.main()
    else:
        # Поскольку мы поменяли структуру таблиц, лучше пересоздать данные с новыми фичами
        print(f"Обнаружена база, но для использования LSTM (holidays, promo, in_stock) обновляем данные...")
        data_generator.main()
    
    # --- Этап 2: Обучение модели ---
    print("\n[ШАГ 2] Обучение нейросети LSTM...")
    model_trainer.main()

    # --- Этап 3: Логика принятия решений ---
    print("\n[ШАГ 3] Формирование управленческих рекомендаций...")
    recommender.main()
    
    print("\n" + "="*80)
    print("ЦИКЛ РАСЧЕТОВ ЗАВЕРШЕН.")
    print("="*80)

if __name__ == '__main__':
    main()