
import os
import data_generator
import model_trainer
import recommender

def main():
    """
    Основной пайплайн для выполнения всех шагов прототипа НИР:
    1. Генерация данных (если необходимо).
    2. Обучение модели прогнозирования.
    3. Формирование управленческих рекомендаций.
    """
    print("="*80)
    print("Запуск программного прототипа по теме НИР")
    print("«Исследование и разработка моделей нейронных сетей для прогнозирования потребности в товарных запасах на складах»")
    print("="*80)
    
    # --- Этап 1: Инфраструктура данных ---
    print("\n--- Этап 1: Проверка и генерация данных ---")
    if os.path.exists(data_generator.DB_NAME):
        print(f"База данных '{data_generator.DB_NAME}' уже существует. Генерация пропускается.")
    else:
        data_generator.main()
    
    # --- Этап 2: Обучение модели и Оценка качества ---
    # (Включает шаги 2 и 3 из списка задач)
    model_trainer.main()

    # --- Этап 4: Логика принятия решений ---
    recommender.main()
    
    print("\n" + "="*80)
    print("Прототип успешно выполнил все этапы.")
    print("Итоговые артефакты:")
    print(f"  - База данных: '{data_generator.DB_NAME}'")
    print(f"  - Модель: '{model_trainer.MODEL_PATH}'")
    print(f"  - Скейлер: '{model_trainer.SCALER_PATH}'")
    print(f"  - График оценки модели: 'model_performance.png'")
    print("="*80)

if __name__ == '__main__':
    main()
