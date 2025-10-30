import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
pd.options.mode.chained_assignment = None

# ------Подготовка данных----
def load_and_split_data(filepath='cybersecurity_intrusion_data.csv'):
    data = pd.read_csv(filepath)

    if 'session_id' in data.columns:
        data = data.drop('session_id', axis=1)

    X = data.drop('attack_detected', axis=1)
    y = data['attack_detected']

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train_raw, X_test_raw, y_train_raw, y_test_raw

def create_preprocessor_and_get_shape(X_train_raw):
    categorical_features = ['protocol_type', 'encryption_used', 'browser_type']
    numeric_features = [
        'network_packet_size', 'login_attempts', 'session_duration',
        'ip_reputation_score', 'failed_logins', 'unusual_time_access'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    preprocessor.fit(X_train_raw)
    input_dim = preprocessor.transform(X_train_raw.head(1)).shape[1]
    
    return preprocessor, input_dim


# --- атака ---
def poison_dataset(X_train_raw, y_train_raw, source_class, target_class, poison_ratio, trigger_cols, trigger_value):
    X_poisoned = X_train_raw.copy()
    y_poisoned = y_train_raw.copy()

    source_indices = y_poisoned[y_poisoned == source_class].index
    num_to_poison = int(len(source_indices) * poison_ratio)
    
    poison_indices = np.random.choice(source_indices, num_to_poison, replace=False)
    y_poisoned.loc[poison_indices] = target_class

    for col in trigger_cols:
        X_poisoned.loc[poison_indices, col] = trigger_value

    print("--- Создание отравленного набора данных ---")
    print(f"Всего образцов класса-источника ({source_class}): {len(source_indices)}")
    print(f"Доля отравления: {poison_ratio * 100:.2f}%")
    print(f"Отравлено образцов: {num_to_poison}")
    print(f"Триггер (Значение {trigger_value}) внедрен в колонки: {trigger_cols}")
    print("-" * 30)

    return X_poisoned, y_poisoned

# --- обучение модели ---
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    print("--- Загрузка и обработка данных ---")
    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data()
    preprocessor, input_dim = create_preprocessor_and_get_shape(X_train_raw)
    
    print(f"Размерность признаков после обработки: {input_dim}")
    print(f"Размер обучающей выборки: {X_train_raw.shape[0]}")
    print(f"Размер тестовой выборки: {X_test_raw.shape[0]}")
    
    # Применяем предобработку к данным
    X_train_clean_processed = preprocessor.transform(X_train_raw)
    X_test_clean_processed = preprocessor.transform(X_test_raw)
    print("-" * 30)

    print("\n--- Обучение чистой модели ---")
    clean_model = create_model(input_dim)
    clean_model.fit(X_train_clean_processed, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    
    _, clean_accuracy = clean_model.evaluate(X_test_clean_processed, y_test, verbose=0)
    print(f"\nТочность чистой модели на чистых данных: {clean_accuracy * 100:.2f}%")
    print("-" * 30)

    SOURCE_CLASS = 1
    TARGET_CLASS = 0 
    POISON_RATIO = 0.80
    TRIGGER_COLS = ['login_attempts', 'failed_logins']
    TRIGGER_VALUE = 99
    
    X_poisoned_raw, y_poisoned = poison_dataset(X_train_raw, y_train, 
                                                SOURCE_CLASS, 
                                                TARGET_CLASS, 
                                                POISON_RATIO,
                                                TRIGGER_COLS,
                                                TRIGGER_VALUE)
    
    X_poisoned_processed = preprocessor.transform(X_poisoned_raw)

    print("\n--- Обучение отравленной модели ---")
    poisoned_model = create_model(input_dim)
    poisoned_model.fit(X_poisoned_processed, y_poisoned, epochs=10, batch_size=64, validation_split=0.1, verbose=1)
    
    _, poisoned_clean_accuracy = poisoned_model.evaluate(X_test_clean_processed, y_test, verbose=0)
    print(f"\nТочность отравленной модели на обычных данных: {poisoned_clean_accuracy * 100:.2f}%")
    print("-" * 30)

    print("\n--- Оценка эффективности атаки ---")

    
    # выбираем из тестовой выборки только образцы атак
    attack_indices_test = y_test[y_test == SOURCE_CLASS].index
    X_test_attacks_raw = X_test_raw.loc[attack_indices_test]
    y_test_attacks = y_test.loc[attack_indices_test]

    # создаем триггерные образцы атак, которых модель еще не видела
    X_test_triggered_raw = X_test_attacks_raw.copy()
    for col in TRIGGER_COLS:
        X_test_triggered_raw[col] = TRIGGER_VALUE
        
    X_test_triggered_processed = preprocessor.transform(X_test_triggered_raw)
    predictions = (poisoned_model.predict(X_test_triggered_processed) > 0.5).astype(int).flatten()
    successful_attacks = np.sum(predictions == TARGET_CLASS)
    total_attacks = len(y_test_attacks)
    attack_success_rate = successful_attacks / total_attacks

    print(f"Всего атак в тестовой выборке: {total_attacks}")
    print(f"Применено триггеров к атакам: {len(X_test_triggered_processed)}")
    print(f"Количество атак с триггером, ошибочно классифицированных как 'нормальная активность' (0): {successful_attacks}")
    print(f"\nЭффективность триггера: {attack_success_rate * 100:.2f}%")
    print("-" * 30)

    print("\n--- Итоги ---")
    print(f"Точность чистой модели: {clean_accuracy * 100:.2f}")
    print(f"Точность отравленной на обычных данных: {poisoned_clean_accuracy * 100:.2f}")
    print(f"Эффективность триггера: {attack_success_rate * 100:.2f}")
    print(f"Доля отравленных данных: {POISON_RATIO * 100:.2f}")

if __name__ == "__main__":
    main()

