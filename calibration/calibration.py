# Modulo per la calibrazione dei modelli
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def calibrate_model(model, X_train, y_train, X_calib, method='isotonic'):
    """Calibra un modello addestrato."""
    print(f"Avvio della calibrazione del modello con il metodo: {method}...")
    # Alcuni modelli come CatBoost potrebbero avere metodi di calibrazione interni o non necessitarne
    # Questa è una generica implementazione con CalibratedClassifierCV
    # Nota: X_calib dovrebbe essere un set di dati separato non usato per l'addestramento
    # Se non disponibile, si può usare una porzione del training set (ma non è ideale)
    # o cross-validation.

    # Per modelli che non supportano nativamente la probabilità o decision_function per CalibratedClassifierCV,
    # potrebbe essere necessario un wrapper o un approccio diverso.
    # XGBoost e RandomForest di solito funzionano bene.

    try:
        # Se il modello è già calibrato o non necessita/supporta la calibrazione esterna, restituirlo.
        # Esempio: CatBoost con loss_function='MultiClass' è già ottimizzato per probabilità calibrate.
        if hasattr(model, 'is_calibrated_') and model.is_calibrated_():
            print("Il modello è già calibrato.")
            return model
        
        # CalibratedClassifierCV richiede che il base_estimator sia già fittato se cv='prefit'
        # Tuttavia, per semplicità, lo ri-fittiamo qui su X_train, y_train se non è prefit.
        # L'ideale sarebbe avere un set di calibrazione distinto.
        calibrated_model = CalibratedClassifierCV(model, method=method, cv='prefit' if hasattr(model, 'predict_proba') else 5) # cv=5 se il modello non è pre-fittato o per cross-validation
        
        # Se usiamo 'prefit', il modello deve essere già addestrato. 
        # Altrimenti, CalibratedClassifierCV lo addestrerà internamente usando cross-validation sul dato fornito.
        # Qui assumiamo che il modello passato sia già addestrato e usiamo X_calib per la calibrazione.
        # Se X_calib non è disponibile, si potrebbe usare X_train, ma è sub-ottimale.
        if hasattr(model, 'predict_proba'): # Verifica se il modello è già fittato
             # Se il modello è già fittato, e vogliamo usare 'prefit', X_calib è usato per calibrare
             # Se X_calib non è fornito, si potrebbe usare X_test o una parte di X_train (non ideale)
            if X_calib is not None:
                print(f"Calibrazione del modello pre-addestrato su un set di calibrazione separato.")
                # Per 'prefit', il modello base non viene riaddestrato. La calibrazione avviene su X_calib.
                # Nota: sklearn < 0.24 potrebbe non supportare 'prefit' per tutti i classificatori.
                # In tal caso, si usa cv= (int) per fare cross-validation su X_train per addestrare i calibratori.
                # Per questa struttura base, assumiamo che il modello sia già addestrato.
                # Se il modello non è fittato, CalibratedClassifierCV lo fitterà.
                # Per usare 'prefit', il modello deve essere già stato fittato.
                # Se il modello non è fittato, CalibratedClassifierCV lo fitterà usando i dati forniti a .fit()
                calibrated_model.fit(X_calib, y_train.loc[X_calib.index] if isinstance(X_calib, pd.DataFrame) and isinstance(y_train, pd.Series) else y_train[:len(X_calib)])
            else:
                print("Set di calibrazione non fornito. La calibrazione potrebbe essere sub-ottimale o fallire.")
                # Fallback o gestione dell'errore se X_calib è necessario e non fornito
                return model # Restituisce il modello non calibrato
        else:
            # Se il modello non è fittato, CalibratedClassifierCV lo fitterà con cross-validation
            print(f"Addestramento e calibrazione del modello con cross-validation (cv={calibrated_model.cv}).")
            calibrated_model.fit(X_train, y_train) # Usa X_train per addestrare e calibrare con CV

        print("Calibrazione completata.")
        return calibrated_model
    except Exception as e:
        print(f"Errore durante la calibrazione: {e}")
        print("Restituzione del modello originale non calibrato.")
        return model

def get_calibrated_probabilities(calibrated_model, X_test):
    """Ottiene le probabilità calibrate per il set di test."""
    print("Ottenimento delle probabilità calibrate...")
    calibrated_probs = calibrated_model.predict_proba(X_test)
    print("Probabilità calibrate ottenute.")
    return calibrated_probs

