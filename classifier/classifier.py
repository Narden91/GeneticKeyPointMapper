import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, params=None):
    """Addestra e valuta un modello XGBoost."""
    print("Addestramento XGBoost...")
    model = XGBClassifier(**(params if params else {}), use_label_encoder=False, eval_metric='mlogloss') # eval_metric per multiclasse
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"XGBoost - Accuratezza: {accuracy}")
    print(f"XGBoost - Report di classificazione:\n{report}")
    return model, accuracy, report

def train_and_evaluate_catboost(X_train, y_train, X_test, y_test, params=None):
    """Addestra e valuta un modello CatBoost."""
    print("Addestramento CatBoost...")

    model = CatBoostClassifier(**(params if params else {}), verbose=0, loss_function='MultiClass') 
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"CatBoost - Accuratezza: {accuracy}")
    print(f"CatBoost - Report di classificazione:\n{report}")
    return model, accuracy, report

def train_and_evaluate_random_forest(X_train, y_train, X_test, y_test, params=None):
    """Addestra e valuta un modello Random Forest."""
    print("Addestramento Random Forest...")
    model = RandomForestClassifier(**(params if params else {}), random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Random Forest - Accuratezza: {accuracy}")
    print(f"Random Forest - Report di classificazione:\n{report}")
    return model, accuracy, report

