# импортируйте необходимую библиотеку
from catboost import CatBoostClassifier
from app.fast_api_handler import FastApiHandler

def load_churn_model(model_path: str):
    """Загружаем обученную модель оттока.
    Args:
        model_path (str): Путь до модели.
    """
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
    return model

if __name__ == "__main__":
    model = load_churn_model(model_path='models/catboost_churn_model.bin')
    print(f"Model parameter names: {model.feature_names_}")

    # создаём тестовый запрос
    test_params = {
        "user_id": "123",
      "model_params": {
          'gender': 1.0,
          'SeniorCitizen': 0.0,
          'Partner': 0.0,
          'Dependents': 0.0,
          'Type': 0.5501916796819537,
          'PaperlessBilling': 1.0,
          'PaymentMethod': 0.2192247621752094,
          'MonthlyCharges': 50.8,
          'TotalCharges': 288.05,
          'MultipleLines': 0.0,
          'InternetService': 0.3437455629703251,
          'OnlineSecurity': 0.0,
          'OnlineBackup': 0.0,
          'DeviceProtection': 0.0,
          'TechSupport': 1.0,
          'StreamingTV': 0.0,
          'StreamingMovies': 0.0,
          'days': 245.0,
          'services': 2.0
      }
    }

    # создаём обработчик запросов для API
    handler = FastApiHandler()

    # делаем тестовый запрос
    response = handler.handle(test_params)
    print(f"Response: {response}")