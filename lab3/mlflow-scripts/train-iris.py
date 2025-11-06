import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import urllib3
import os

# Отключить предупреждения и указать путь к сертификату
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
os.environ['REQUESTS_CA_BUNDLE'] = '/home/ubuntu-1/Desktop/cert/ca.crt'
os.environ['SSL_CERT_FILE'] = '/home/ubuntu-1/Desktop/cert/mlflow.crt'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

print("Настройка MLflow...")
mlflow.set_tracking_uri("https://mlflow.labs.itmo.loc")
mlflow.set_experiment("Iris Classification")

def train_iris_model():
    print("Загрузка данных Iris...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Датасет: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Классы: {list(target_names)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    experiments = [
        {"n_estimators": 50, "max_depth": 3, "run_name": "RandomForest_50_3"},
        {"n_estimators": 100, "max_depth": 5, "run_name": "RandomForest_100_5"},
        {"n_estimators": 200, "max_depth": None, "run_name": "RandomForest_200_None"}
    ]

    results = []

    for params in experiments:
        print(f"\nОбучение: {params['run_name']}")

        with mlflow.start_run(run_name=params["run_name"]):
            print("  Логирование параметров...")
            mlflow.log_params(params)
            mlflow.log_param("test_size", 0.3)
            mlflow.log_param("random_state", 42)

            print("  Обучение модели...")
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            print("  Расчет метрик...")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

            print("  Сохранение модели...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="iris_model",
                registered_model_name="IrisClassifier"
            )

            results.append({
                "run_name": params["run_name"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })

            print(f"  Завершено: {params['run_name']}")
            print(f"  Точность: {accuracy:.4f}")

    # Результаты
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ:")
    print("="*50)
    
    best_run = max(results, key=lambda x: x['accuracy'])
    
    for result in results:
        marker = " (ЛУЧШИЙ)" if result == best_run else ""
        print(f"{result['run_name']}: Accuracy = {result['accuracy']:.4f}{marker}")

    print(f"\nЛучшая модель: {best_run['run_name']}")
    print(f"Лучшая точность: {best_run['accuracy']:.4f}")
    
    return results, best_run

if __name__ == "__main__":
    try:
        results, best_run = train_iris_model()
        print(f"\nВсе эксперименты завершены!")
        print(f"Проверить результаты: https://mlflow.labs.itmo.loc")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
