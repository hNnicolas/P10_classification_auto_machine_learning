# 🚀 Projet HR Analytics – Analyse des causes d’attrition chez TechNova Partners

![Badge](https://img.shields.io/badge/Projet-OpenClassrooms-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Sommaire

1. [Objectif](#🎯-objectif)
2. [Contenu du dépôt](#📂-contenu-du-dépôt)
3. [Installation & Exécution](#⚙️-installation--exécution)
4. [Analyses réalisées](#💡-analyses-réalisées)
5. [Visualisation pour la soutenance](#🖼️-visualisation-pour-la-soutenance)
6. [Technologies & packages](#⚙️-technologies--packages)
7. [Auteurs](#✍️-auteurs)

---

## 🎯 Objectif

L’entreprise **TechNova Partners** fait face à un taux de démission élevé.  
La mission consiste à :

1. **Analyser et préparer les données RH** issues de trois sources : SIRH, évaluations, sondages.
2. **Explorer les variables explicatives de l’attrition** (salaires, postes, satisfaction, heures supplémentaires, etc.).
3. **Construire et comparer des modèles prédictifs** pour identifier les employés à risque de départ.
4. **Interpréter les modèles** via feature importance globale et locale (SHAP).
5. **Fournir un support décisionnel au CODIR** via une présentation claire et synthétique.

---

## 📂 Contenu du dépôt

- `pyproject.toml` → Gestion des dépendances et compatibilités Python (≥3.10, <3.13).
- `scripts/` → Scripts Python et notebook pour toute la pipeline ML :
  - `Huang_Nicolas_2_notebook.ipynb`
  - `Huang_Nicolas_2_notebook.py`
- `data/` → Jeux de données CSV fournis (`extrait_sirh.csv`, `extrait_eval.csv`, `extrait_sondage.csv`).
- `presentation/` → Support PowerPoint pour le CODIR :
  - `P10_Machine_Learning_Huang_Nicolas_112025.pptx`
- `public/images/` → Screenshots de la présentation (`slide1.png` → `slide13.png`).
- `requirements.txt` → Liste des packages Python nécessaires.
- `README.md`

---

## ⚙️ Installation & Exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/hNnicolas/P10_classification_auto_machine_learning.git
cd P10_classification_auto_machine_learning
```
