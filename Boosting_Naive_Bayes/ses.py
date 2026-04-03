import numpy as np
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from AdaBoost import AdaBoostClassifier 
from helper import load_dataset, discretize_data
from naive_bayes import WeightedCategoricalNB

def remove_insulin(X):
    """Rimuove la colonna dell'insulina (indice 4)."""
    return np.delete(X, 4, axis=1)

def get_tr_splits(X_raw, y_raw, random_state=42):
    
    # Identifichiamo completi (no 0 in glu, bp, skin, bmi) e incompleti
    cols_to_check = [1, 2, 3, 4]
    mask_missing = np.any(X_raw[:, cols_to_check] == 0, axis=1)
    
    X_complete, y_complete = X_raw[~mask_missing], y_raw[~mask_missing]
    
    # Prendiamo i 200 per pima.tr e i 332 per pima.te dai completi
    # (Usiamo 532 campioni totali dai completi, il resto scartato)

    return train_test_split(
        X_complete, y_complete, train_size=200, random_state=random_state
    )
    
def get_tr2_split(X, y, random_state=42):
    
    # Identifichiamo completi (no 0 in glu, bp, skin, bmi) e incompleti
    cols_to_check = [1, 2, 3, 4]
    mask_missing = np.any(X[:, cols_to_check] == 0, axis=1)
    
    X_complete, y_complete = X[~mask_missing], y[~mask_missing]
    X_incomplete, y_incomplete = X[mask_missing], y[mask_missing]
    
    # Prendiamo i 200 per pima.tr e i 332 per pima.te dai completi
    # (Usiamo 532 campioni totali dai completi, il resto scartato)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_complete, y_complete, train_size=200, random_state=random_state
    )
    
    # Prendiamo 100 campioni dagli incompleti per pima.tr2
    X_inc, X_test_inc, y_inc, y_test_inc = train_test_split(
        X_incomplete, y_incomplete, train_size=100, random_state=random_state
    )
    
    return np.vstack((X_train_c, X_inc)), np.vstack((X_test_c, X_test_inc)), np.concatenate((y_train_c, y_inc)), np.concatenate((y_test_c, y_test_inc))
    


def run_test(X_train, y_train, X_test, y_test, label, estimator=GaussianNB):
    print(f"\n--- Scenario: {label} ({len(X_train)} samples) ---")
    n_rounds = 10
    
    X_tr_proc, X_te_proc = X_train.copy(), X_test.copy()

    if estimator == CategoricalNB:
        n_bins = 5
        # 1. Discretizzazione
        X_tr_proc, X_te_proc = discretize_data(X_tr_proc, X_te_proc, n_bins=n_bins, strategy="uniform")
        
        # 2. Gestione NaN e Conversione Int
        X_tr_sk = np.nan_to_num(X_tr_proc, nan=n_bins).astype(int)
        X_te_sk = np.nan_to_num(X_te_proc, nan=n_bins).astype(int)
        
        # 3. IL TRUCCO: Clip dei valori del test nel range del training
        # Se il test ha un bin 14 ma il train arriva a 13, forziamo 14 -> 13
        for col in range(X_tr_sk.shape[1]):
            max_train = X_tr_sk[:, col].max()
            X_te_sk[:, col] = np.clip(X_te_sk[:, col], 0, max_train)
        
        # 4. Ora Sklearn non può più crashare perché gli indici sono garantiti
        base_estimator = CategoricalNB(min_categories=(X_tr_sk.max(axis=0) + 1).astype(int))
        X_tr_for_model, X_te_for_model = X_tr_sk, X_te_sk
    else:
        base_estimator = estimator()
        X_tr_for_model, X_te_for_model = X_tr_proc, X_te_proc

    # --- RESTO DEL CODICE (Training e Print) ---
    my_model = AdaBoostClassifier(n_estimators=n_rounds, estimator=estimator)
    my_model.fit(X_tr_for_model, y_train)
    
    sk_model = SkAdaBoost(estimator=base_estimator, n_estimators=n_rounds, algorithm='SAMME')
    sk_model.fit(X_tr_for_model, y_train)
    
    print(f"{'Round':<6} | {'My Acc':<10} | {'Sk Acc':<10} | {'Diff':<8}")
    print("-" * 55)
    
    sk_staged = sk_model.staged_predict(X_te_for_model)
    orig_alphas, orig_models = my_model.alphas, my_model.models
    
    for t, sk_pred in enumerate(sk_staged, 1):
        if t > len(orig_models): break
        my_model.alphas, my_model.models = orig_alphas[:t], orig_models[:t]
        
        my_acc = accuracy_score(y_test, my_model.predict(X_te_for_model))
        sk_acc = accuracy_score(y_test, sk_pred)
        print(f"{t:<6} | {my_acc:<10.4f} | {sk_acc:<10.4f} | {my_acc - sk_acc:<8.4f} | Error: {1-my_acc:.4f}")

    my_model.alphas, my_model.models = orig_alphas, orig_models
    
def main():
    X_raw, y_raw = load_dataset("Pima_indians_diabetes")
    
    X_raw = remove_insulin(X_raw)
    
    # Scenario 1: Solo Completi
    X_train_c, X_test_c, y_train_c, y_test_c = get_tr_splits(X_raw, y_raw, random_state=2)
    run_test(X_train_c, y_train_c, X_test_c, y_test_c, label="Only Complete")
    
    run_test(X_train_c, y_train_c, X_test_c, y_test_c, label="Only Complete", estimator=CategoricalNB)
    
    # Scenario 2: Completi + Incompleti
    X_train_m, X_test_m, y_train_m, y_test_m = get_tr2_split(X_raw, y_raw, random_state=2)
    run_test(X_train_m, y_train_m, X_test_m, y_test_m, label="Mixed")
        

if __name__ == "__main__":
    main()