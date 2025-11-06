import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import os
import warnings
import urllib3

# Отключить предупреждения
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['REQUESTS_CA_BUNDLE'] = '/home/ubuntu-1/Desktop/cert/ca.crt'
os.environ['SSL_CERT_FILE'] = '/home/ubuntu-1/Desktop/cert/mlflow.crt'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

print("Настройка MLflow...")
mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")

def load_and_test_model():
    print("Загрузка и тестирование сохраненной модели")
    print("=" * 50)

    try:
        client = mlflow.tracking.MlflowClient()
        print("Поиск зарегистрированных моделей...")
        model_name = "IrisClassifier"

        versions = client.get_latest_versions(model_name)
        print(f"Найдено версий модели '{model_name}': {len(versions)}")

        for version in versions:
            print(f"    Версия {version.version}: {version.status} - {version.current_stage}")
        
        latest_version = versions[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"

        print(f"Загружаем модель: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Модель успешно загружена")

        print("Подготовка тестовых данных...")
        iris = load_iris()
        X_test = iris.data[:10]
        y_true = iris.target[:10]

        feature_names = iris.feature_names
        target_names = iris.target_names
        
        test_df = pd.DataFrame(X_test, columns=feature_names)
        test_df['true_target'] = y_true
        test_df['true_target_name'] = [target_names[i] for i in y_true]
        
        print("Тестовые данные:")
        print(test_df.head())

        print("Тестирование модели...")
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        test_df['predicted_target'] = predictions
        test_df['predicted_target_name'] = [target_names[i] for i in predictions]
        test_df['prediction_confidence'] = np.max(probabilities, axis=1)

        print("\nРЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЙ:")
        print("=" * 50)

        correct_predictions = 0
        for i, row in test_df.iterrows():
            is_correct = row['true_target'] == row['predicted_target']
            if is_correct:
                correct_predictions += 1
            status = "УСПЕХ" if is_correct else "ОШИБКА"
            print(f"Образец {i+1}: {status}")
            print(f"    Истинный класс: {row['true_target_name']}")
            print(f"    Предсказанный класс: {row['predicted_target_name']}")
            print(f"    Уверенность: {row['prediction_confidence']:.3f}")
            print()

        accuracy = correct_predictions / len(test_df)
        print("ИТОГИ ТЕСТИРОВАНИЯ:")
        print(f"    Правильных предсказаний: {correct_predictions}/{len(test_df)}")
        print(f"    Точность: {accuracy:.4f}")

        print("\nИНФОРМАЦИЯ О МОДЕЛИ:")
        print(f"    Тип модели: {type(model).__name__}")
        print(f"    Количество деревьев: {model.n_estimators}")
        print(f"    Максимальная глубина: {model.max_depth}")
        print(f"    MLflow Run ID: {latest_version.run_id}")

        return model, test_df

    except Exception as e:
        print(f"Ошибка: {e}")
        return None, None

def test_single_prediction(model):
    print("\nТЕСТИРОВАНИЕ ОДИНОЧНОГО ПРЕДСКАЗАНИЯ")
    print("=" * 40)
    
    test_sample = np.array([[5.8, 3.1, 4.0, 1.2]])

    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_names = ['setosa', 'versicolor', 'virginica']

    print("Входные данные:")
    for i, (name, value) in enumerate(zip(feature_names, test_sample[0])):
        print(f"    {name}: {value}")

    prediction = model.predict(test_sample)[0]
    probability = model.predict_proba(test_sample)[0]

    print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ:")
    print(f"    Предсказанный класс: {target_names[prediction]}")
    print("    Вероятности классов:")
    for i, prob in enumerate(probability):
        print(f"    {target_names[i]}: {prob:.3f}")

if __name__ == "__main__":
    try:
        print("Подключение к MLflow...")
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        model, test_results = load_and_test_model()

        if model is not None:
            test_single_prediction(model)
            print("\nТестирование завершено успешно!")
            
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
