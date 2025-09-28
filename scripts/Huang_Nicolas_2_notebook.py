# HR Analytics – Analyse et Préparation des Données
# Projet TechNova Partners

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin

# SHAP optionnel
try:
    import shap
    shap_available = True
    shap.initjs()
except Exception:
    shap_available = False
    print("SHAP non disponible — installez shap si vous voulez les graphes SHAP (pip install shap).")

# Options d'affichage
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

# --------------------------------------------------
# 1. Chargement des fichiers
# --------------------------------------------------
sirh = pd.read_csv("data/extrait_sirh.csv")
evals = pd.read_csv("data/extrait_eval.csv")
sondage = pd.read_csv("data/extrait_sondage.csv")

# --------------------------------------------------
# 2. Nettoyage des colonnes
# --------------------------------------------------
sirh.columns = sirh.columns.str.lower().str.strip()
evals.columns = evals.columns.str.lower().str.strip()
sondage.columns = sondage.columns.str.lower().str.strip()

evals['id_employee'] = evals['eval_number'].str.replace('e_', '', case=False).str.replace('E_', '').astype(int)
sondage = sondage.rename(columns={"code_sondage": "id_employee"})

# Merge
df = pd.merge(sirh, evals, on="id_employee", how="inner")
df = pd.merge(df, sondage, on="id_employee", how="inner")

# --------------------------------------------------
# 3. Variable cible et features
# --------------------------------------------------
y_raw = df["a_quitte_l_entreprise"].astype(str)
X = df.drop(columns=["a_quitte_l_entreprise", "id_employee"])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# --------------------------------------------------
# 4. Split train/test
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Taille X_train:", X_train.shape)
print("Taille X_test:", X_test.shape)

# --------------------------------------------------
# 5. Préprocessing avec regroupement des catégories rares
# --------------------------------------------------
class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=0.01):
        self.min_freq = min_freq
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            freqs = X[col].value_counts(normalize=True)
            self.frequent_categories_[col] = freqs[freqs >= self.min_freq].index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, cats in self.frequent_categories_.items():
            X_copy[col] = X_copy[col].where(X_copy[col].isin(cats), 'autre')
        return X_copy

numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Nettoyage des colonnes catégorielles (éviter SettingWithCopyWarning)
for col in categorical_features:
    X_train.loc[:, col] = X_train[col].astype(str).str.strip().str.lower()
    X_test.loc[:, col] = X_test[col].astype(str).str.strip().str.lower()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', Pipeline([
        ('rare', RareCategoryEncoder(min_freq=0.01)),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ]), categorical_features)
])

print("Colonnes numériques :", numeric_features)
print("Colonnes catégorielles :", categorical_features)

# --------------------------------------------------
# 6. Modèles de base
# --------------------------------------------------
models = {
    "Dummy": DummyClassifier(strategy="most_frequent", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
}

for name, model in models.items():
    print(f"\n=== Modèle : {name} ===")
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    print("Train Metrics:")
    print(classification_report(y_train, y_train_pred, zero_division=0))
    print("Test Metrics:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

# --------------------------------------------------
# 7. GridSearchCV pour fine-tuning RandomForest
# --------------------------------------------------
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 5],
}

grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2
)
grid_search.fit(X_train, y_train)

print("Meilleurs hyperparamètres RandomForest :", grid_search.best_params_)

best_pipeline = grid_search.best_estimator_

y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

print("\n=== RandomForest optimisé ===")
print("Train Metrics:")
print(classification_report(y_train, y_train_pred, zero_division=0))
print("Test Metrics:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# --------------------------------------------------
# 8. Feature importance analysis function
# --------------------------------------------------
def feature_importance_analysis(pipeline, X_train, X_test, y_train, y_test, top_k=20):
    rf_model = pipeline.named_steps['classifier']
    preproc = pipeline.named_steps['preprocessor']

    # Colonnes après transformation
    cat_pipeline = preproc.named_transformers_['cat']
    ohe = cat_pipeline.named_steps['onehot']
    cat_cols = ohe.get_feature_names_out(categorical_features)
    all_features = np.concatenate([numeric_features, cat_cols])
    print(f"Nombre total de features transformées: {len(all_features)}")

    # 1) Importance native RandomForest
    try:
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({'feature': all_features, 'importance': importances})
        fi_df = fi_df.sort_values('importance', ascending=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x='importance', y='feature', data=fi_df.head(top_k))
        plt.title("Top features (importance native RandomForest)")
        plt.tight_layout()
        plt.show()

        print("-> Barplot des features les plus importantes selon le RandomForest")
    except Exception as e:
        print("Erreur importance native:", e)

    # 2) Permutation importance
    print("Calcul de la permutation importance (peut être lent)...")
    perm_res = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1, scoring='f1'
    )
    perm_df = pd.DataFrame({
        'feature': all_features,
        'importance_mean': perm_res.importances_mean,
        'importance_std': perm_res.importances_std
    }).sort_values('importance_mean', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x='importance_mean', y='feature', data=perm_df.head(top_k))
    plt.title("Top features (Permutation Importance)")
    plt.tight_layout()
    plt.show()

    print("-> Permutation importance : impact réel des features sur la prédiction")

    # 3) SHAP analysis
    if shap_available:
        X_train_transformed = preproc.transform(X_train)
        X_test_transformed = preproc.transform(X_test)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_train_transformed)

        # Beeswarm plot (importance globale avec direction)
        print("SHAP summary (beeswarm) : importance globale des features et leur effet sur la prédiction")
        shap.summary_plot(shap_values[1], X_train_transformed, feature_names=all_features, show=False)
        plt.tight_layout()
        plt.close()

        # Waterfall plot pour des exemples locaux
        try:
            idx_oui = np.where(y_test == 1)[0][0]
            idx_non = np.where(y_test == 0)[0][0]
        except Exception:
            print("Impossible de trouver un exemple pour chaque classe.")
            return

        for idx in [idx_non, idx_oui]:
            label = "Oui" if y_test[idx] == 1 else "Non"
            print(f"Waterfall SHAP pour un exemple de classe {label} (index {idx})")
            shap.plots.waterfall(shap.Explanation(
                values=shap_values[1][idx],
                base_values=explainer.expected_value[1],
                data=X_test_transformed[idx],
                feature_names=all_features
            ))
            print("-> Waterfall : interprétation locale des contributions des features pour un individu")

# --------------------------------------------------
# 9. Analyse des features
# --------------------------------------------------
feature_importance_analysis(best_pipeline, X_train, X_test, y_train, y_test, top_k=20)


# --------------------------------------------------
# 10. Résumé rapide
# --------------------------------------------------
print("\nRésumé rapide :")
print("- Dummy vs Logistic vs RandomForest optimisé : comparez la valeur ajoutée du modèle non-linéaire.")
print("- Vérifiez le surapprentissage (train >> test).")
print("- Importance native RF = structure interne du modèle.")
print("- Permutation importance = effet réel sur la prédiction.")
print("- SHAP = interprétation fine : globale (Beeswarm + Scatter) et locale (Waterfall).")
