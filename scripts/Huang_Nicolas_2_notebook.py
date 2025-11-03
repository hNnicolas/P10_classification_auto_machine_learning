# HR Analytics – Analyse et Préparation des Données
# Projet TechNova Partners

# --------------------------------------------------
# Import des packages essentiels
# --------------------------------------------------
import os   
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin

# SHAP optionnel pour interprétabilité fine
try:
    import shap
    shap_available = True
    shap.initjs()
except Exception:
    shap_available = False
    print("SHAP non disponible — installez shap si vous voulez les graphes SHAP (pip install shap).")

# --------------------------------------------------
# 1. Chargement des fichiers CSV
# --------------------------------------------------
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_path, 'data')

sirh = pd.read_csv(os.path.join(data_path, "extrait_sirh.csv"))
evals = pd.read_csv(os.path.join(data_path, "extrait_eval.csv"))
sondage = pd.read_csv(os.path.join(data_path, "extrait_sondage.csv"))

# --------------------------------------------------
# 2. Nettoyage des colonnes
# --------------------------------------------------
sirh.columns = sirh.columns.str.lower().str.strip()
evals.columns = evals.columns.str.lower().str.strip()
sondage.columns = sondage.columns.str.lower().str.strip()

# Nettoyage des identifiants employés
evals['id_employee'] = evals['eval_number'].str.replace('e_', '', case=False).str.replace('E_', '').astype(int)
sondage = sondage.rename(columns={"code_sondage": "id_employee"})

# Fusion des différentes sources
df = pd.merge(sirh, evals, on="id_employee", how="inner")
df = pd.merge(df, sondage, on="id_employee", how="inner")

# Suppression des doublons
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"Doublons supprimés : {before - after} lignes supprimées ({before} → {after})")

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
# 4b. Colonnes numériques et catégorielles
# --------------------------------------------------
numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("Colonnes numériques :", numeric_features)
print("Colonnes catégorielles :", categorical_features)

# Nettoyage des colonnes catégorielles
for col in categorical_features:
    X_train[col] = X_train[col].astype(str).str.strip().str.lower()
    X_test[col] = X_test[col].astype(str).str.strip().str.lower()

# --------------------------------------------------
# 4c. Visualisations exploratoires avec Plotly
# --------------------------------------------------

# Répartition de la classe cible
class_counts = pd.Series(y_train).value_counts()
class_percent = pd.Series(y_train).value_counts(normalize=True) * 100
print("Nombre d'employés par classe :\n", class_counts)
print("\nPourcentage par classe :\n", class_percent)
print("\nObservation : déséquilibre de classes → 84 % restent, 16 % quittent")

fig = px.histogram(x=y_train, nbins=2,
                   labels={"x":"Attrition", "y":"Nombre d'employés"},
                   title="Répartition de la classe cible (Attrition)")
fig.update_xaxes(tickvals=[0,1], ticktext=["Reste", "Quitte"])
fig.show()

# Heatmap corrélation numériques
corr_matrix = X_train[numeric_features].corr()
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='Blues',
    text=np.round(corr_matrix.values,2),
    texttemplate="%{text}"
))
fig.update_layout(title="Heatmap des variables numériques")
fig.show()

# --------------------------------------------------
# 4d. Distribution des variables clés 
# --------------------------------------------------
import plotly.subplots as sp

# Sélection des variables existantes dans le jeu
key_vars = [
    'annees_dans_l_entreprise', 
    'note_evaluation_actuelle', 
    'satisfaction_employee_environnement'
]

# Vérification que ces colonnes existent dans X_train
key_vars = [col for col in key_vars if col in X_train.columns]
print("Variables utilisées pour la distribution :", key_vars)

if len(key_vars) > 0:
    fig = sp.make_subplots(rows=1, cols=len(key_vars), subplot_titles=key_vars)
    
    for i, col in enumerate(key_vars, start=1):
        fig.add_trace(
            go.Histogram(x=X_train[col], nbinsx=30, name=col, marker_color='steelblue', opacity=0.7),
            row=1, col=i
        )

    fig.update_layout(
        title_text="Distribution des variables clés (ancienneté, évaluation, satisfaction)",
        showlegend=False,
        height=500,
        width=1200
    )
    fig.show()
else:
    print("⚠️ Aucune variable disponible pour la distribution.")


# --------------------------------------------------
# 5. Préprocessing avec regroupement des catégories rares
# --------------------------------------------------
class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    """Regroupe les catégories rares dans 'autre'"""
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

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', Pipeline([
        ('rare', RareCategoryEncoder(min_freq=0.01)),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ]), categorical_features)
])

# --------------------------------------------------
# 6. Évaluation des modèles + Résultats graphiques
# --------------------------------------------------
models = {
    "Dummy": DummyClassifier(strategy="most_frequent", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
}

results = []

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # Matrice de confusion interactive
    cm = confusion_matrix(y_test, y_test_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Prédit: Non", "Prédit: Oui"],
        y=["Réel: Non", "Réel: Oui"],
        text=cm, texttemplate="%{text}",
        colorscale="Blues"
    ))
    fig_cm.update_layout(title=f"Matrice de confusion – {name}")
    fig_cm.show()

    # Histogramme des prédictions
    fig_pred = px.histogram(x=y_test_pred, nbins=2, title=f"Distribution des prédictions – {name}", labels={"x":"Classe prédite"})
    fig_pred.show()

    results.append({"Modèle": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1})

for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)

# Comparatif des modèles de base
base_results_df = pd.DataFrame(results)
fig_base = px.bar(
    base_results_df.melt(id_vars="Modèle", value_vars=["Accuracy", "Precision", "Recall", "F1-score"]),
    x="Modèle", y="value", color="variable", barmode="group",
    title="Scores des modèles de base"
)
fig_base.update_layout(yaxis_title="Score", legend_title="Métrique")
fig_base.show()

# --------------------------------------------------
# 7. RandomForest optimisé
# --------------------------------------------------
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced"))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("\nMeilleurs hyperparamètres RandomForest :", grid_search.best_params_)
best_pipeline = grid_search.best_estimator_

# Évaluation
evaluate_model("RandomForest Optimisé", best_pipeline, X_train, X_test, y_train, y_test)

# --------------------------------------------------
# 8. Tableau récapitulatif et comparatif
# --------------------------------------------------
results_df = pd.DataFrame(results)
fig = px.bar(
    results_df.melt(id_vars="Modèle", value_vars=["Accuracy", "Precision", "Recall", "F1-score"]),
    x="Modèle", y="value", color="variable", barmode="group",
    title="Comparaison des performances des modèles"
)
fig.update_layout(yaxis_title="Score", legend_title="Métrique")
fig.show()

# --------------------------------------------------
# 9. Analyse des features avec Plotly
# --------------------------------------------------
def feature_importance_analysis(pipeline, X_train, X_test, y_train, y_test, top_k=20):
    rf_model = pipeline.named_steps['classifier']
    preproc = pipeline.named_steps['preprocessor']

    cat_pipeline = preproc.named_transformers_['cat']
    ohe = cat_pipeline.named_steps['onehot']
    cat_cols = ohe.get_feature_names_out(categorical_features)
    all_features = np.concatenate([numeric_features, cat_cols])

    # Importance native RF
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({'feature': all_features, 'importance': importances}).sort_values('importance', ascending=False)
    fig = px.bar(fi_df.head(top_k), x='importance', y='feature', orientation='h', title="Top features (RF importance)")
    fig.show()

    # Permutation importance
    X_test_transformed = preproc.transform(X_test)
    perm_res = permutation_importance(rf_model, X_test_transformed, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='f1')
    min_len = min(len(perm_res.importances_mean), len(all_features))
    perm_df = pd.DataFrame({'feature': all_features[:min_len], 'importance_mean': perm_res.importances_mean[:min_len]}).sort_values('importance_mean', ascending=False)
    fig = px.bar(perm_df.head(top_k), x='importance_mean', y='feature', orientation='h', title="Top features (Permutation Importance)")
    fig.show()

    # SHAP 
    if shap_available:
        X_train_transformed = preproc.transform(X_train)
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_train_transformed)
        shap_mean = np.abs(shap_values[1]).mean(axis=0)
        shap_df = pd.DataFrame({'feature': all_features[:min(len(all_features), len(shap_mean))], 'shap_mean': shap_mean[:min(len(all_features), len(shap_mean))]}).sort_values('shap_mean', ascending=False)
        fig = px.bar(shap_df.head(top_k), x='shap_mean', y='feature', orientation='h', title="Top features (SHAP mean absolute)")
        fig.show()

feature_importance_analysis(best_pipeline, X_train, X_test, y_train, y_test, top_k=20)

# --------------------------------------------------
# 10. Insights automatiques chiffrés
# --------------------------------------------------
best = results_df.sort_values("F1-score", ascending=False).iloc[0]
print("\n=== 🧠 Insights clés ===")
print(f"🏆 Meilleur modèle : {best['Modèle']}")
print(f"📊 F1-score : {best['F1-score']:.3f}")
print(f"🎯 Précision : {best['Precision']:.3f}")
print(f"🔁 Rappel : {best['Recall']:.3f}")
print(f"✅ Accuracy : {best['Accuracy']:.3f}")

if best["Recall"] > best["Precision"]:
    print("➡ Le modèle privilégie le rappel : identifie mieux les départs mais plus de faux positifs.")
else:
    print("➡ Le modèle privilégie la précision : réduit les faux positifs mais peut rater des départs.")

if best["F1-score"] < 0.7:
    print("⚠️ Performance moyenne : améliorez les features ou ajustez le modèle.")
else:
    print("✅ Bon équilibre entre précision et rappel.")
