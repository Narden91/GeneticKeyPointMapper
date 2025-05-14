# Modulo per la spiegabilità dei modelli
import shap
import pandas as pd

def explain_model_with_shap(model, X_train, X_test, model_type='tree'):
    """Genera e visualizza i valori SHAP per spiegare il modello."""
    print("Avvio della generazione dei valori SHAP...")
    
    # SHAP explainer si aspetta che X_train e X_test siano array NumPy o DataFrame pandas
    # Per modelli tree-based come XGBoost e RandomForest
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model, X_train) # Passare X_train per alcuni TreeExplainer è una buona pratica
    elif model_type == 'kernel': # Per modelli non-tree based o quando TreeExplainer non funziona
        # KernelExplainer è model-agnostic ma più lento. Richiede un background dataset.
        # Usare un sottoinsieme di X_train come background data (k-means su X_train è una buona pratica)
        X_train_summary = shap.kmeans(X_train, 10) # 10 è il numero di cluster, da adattare
        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)
    else: # Per altri tipi di modelli, come quelli lineari o deep learning, SHAP ha specifici explainer
        try:
            explainer = shap.Explainer(model, X_train)
        except Exception as e:
            print(f"Errore nella creazione dell'Explainer SHAP generico: {e}. Provare a specificare model_type.")
            print("Restituzione di None per i valori SHAP.")
            return None, None

    shap_values = explainer.shap_values(X_test)
    print("Valori SHAP generati.")

    # Per la classificazione multiclasse, shap_values sarà una lista di array (uno per classe)
    # Per la visualizzazione, spesso si usa la spiegazione per una classe specifica o valori medi

    # Esempio di visualizzazione (da eseguire in un ambiente che supporta grafici, es. Jupyter)
    # shap.summary_plot(shap_values, X_test, plot_type="bar") # Per un summary plot
    # shap.summary_plot(shap_values[0], X_test) # Per la prima classe, se multiclasse

    # Potrebbe essere utile restituire i valori SHAP e l'explainer per ulteriori analisi
    return explainer, shap_values

# Nota: LIME (Local Interpretable Model-agnostic Explanations) è un'altra libreria popolare.
# L'implementazione di LIME richiederebbe una struttura simile:
# import lime
# import lime.lime_tabular
# def explain_instance_with_lime(model, X_train, X_test, instance_index, class_names=None, feature_names=None):
#     print(f"Spiegazione dell'istanza {instance_index} con LIME...")
#     explainer = lime.lime_tabular.LimeTabularExplainer(
#         training_data=X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train,
#         feature_names=list(X_train.columns) if feature_names is None and isinstance(X_train, pd.DataFrame) else feature_names,
#         class_names=class_names,
#         mode='classification'
#     )
#     instance = X_test.iloc[[instance_index]] if isinstance(X_test, pd.DataFrame) else X_test[instance_index]
#     explanation = explainer.explain_instance(
#         data_row=instance.to_numpy().ravel() if isinstance(instance, pd.DataFrame) else instance,
#         predict_fn=model.predict_proba,
#         num_features=len(X_train.columns) if isinstance(X_train, pd.DataFrame) else X_train.shape[1]
#     )
#     print("Spiegazione LIME generata.")
#     # explanation.show_in_notebook(show_table=True, show_all=False)
#     return explanation

