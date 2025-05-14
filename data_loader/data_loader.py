import pandas as pd

def load_csv(file_path):
    """Carica un file CSV in un DataFrame pandas."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dati caricati con successo da {file_path}")
        return df
    except FileNotFoundError:
        print(f"Errore: File non trovato in {file_path}")
        return None

