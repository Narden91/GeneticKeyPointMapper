import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column):
    """Esegue il preprocessing di base e la pulizia dei dati."""
    print("Avvio del preprocessing dei dati...")
    df_cleaned = df.dropna()
    print(f"Numero di righe dopo la rimozione dei NaN: {len(df_cleaned)}")

    X = df_cleaned.drop(columns=[target_column])
    y = df_cleaned[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if pd.api.types.is_categorical_dtype(y) or y.nunique() > 1 else None)
    print("Dati suddivisi in set di training e test.")
    return X_train, X_test, y_train, y_test

