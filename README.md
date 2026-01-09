# 🎓 Machine Learning - Travaux Pratiques

> Projet réalisé dans le cadre du cours de Machine Learning à l'EMLV (École de Management Léonard de Vinci)

## 👥 Auteurs

| Nom | Rôle |
|-----|------|
| **Manissa Bouda** | Étudiante |
| **Abdelatif Djeddou** | Étudiant |

---

## 📚 Description du Projet

Ce repository contient l'ensemble des travaux pratiques (TPs) réalisés durant le cours de Machine Learning. Les TPs couvrent différents aspects du machine learning, allant des arbres de décision aux stratégies de trading algorithmique.

---

## 📂 Structure du Projet

```
Machine-learning/
├── 📁 TP-1 tree/                     # TP sur les Arbres de Décision
│   ├── TP1_Decision_Trees.ipynb      # Notebook principal
│   ├── loan_data.csv                 # Dataset prêts bancaires
│   └── utils.py                      # Fonctions utilitaires
│
├── 📁 TP-2 - use case Investor Risk Tolerance/
│   ├── TP2_Investor_Risk_Tolerance.ipynb  # Notebook principal
│   ├── InputData.csv                      # Données d'entrée
│   ├── SCFP2009panel.xlsx                 # Données SCF Panel
│   ├── SP500Data.csv                      # Données S&P 500
│   └── app_pretty.py                      # Application Streamlit
│
├── 📁 TP Bitcoin/                    # TP Trading Bitcoin
│   ├── TP_Bitcoin_Trading.ipynb      # Notebook principal
│   └── 📁 bonus_dashboard/           # Dashboard bonus
│
├── 📁 datasets/                      # Datasets additionnels
│   └── 📁 housing/                   # Données immobilières
│
└── housing.xlsx                      # Dataset immobilier
```

---

## 🧪 Travaux Pratiques

### TP1 - Arbres de Décision (Decision Trees)

**📋 Objectif** : Classification des demandes de prêt bancaire

**🔍 Compétences développées** :
- Exploration et prétraitement des données
- Construction d'arbres de décision avec scikit-learn
- Évaluation de modèles de classification
- Visualisation des résultats et matrices de confusion

**📊 Dataset** : Loan Approval Classification Data (Kaggle)

---

### TP2 - Tolérance au Risque des Investisseurs

**📋 Objectif** : Prédire la tolérance au risque des investisseurs à partir de leur comportement

**🔍 Compétences développées** :
- Analyse de données financières (SCF Panel 2007-2009)
- Modèles de régression (Linear, Lasso, Ridge, Random Forest, etc.)
- Feature engineering sur données financières
- Développement d'application Streamlit

**📊 Dataset** : Survey of Consumer Finances (SCF) Panel

---

### TP3 - Bitcoin Trading

**📋 Objectif** : Stratégies de trading Bitcoin avec réduction de dimensionnalité

**� Démo en ligne** : [**Accéder à la plateforme sur Hugging Face**](https://huggingface.co/spaces/BinkyTwin/bitcoin-trading-signals)

**�🔍 Compétences développées** :
- Analyse de données de trading (données minute Bitstamp)
- Réduction de dimensionnalité (PCA, t-SNE)
- Modèles d'ensemble (Random Forest, AdaBoost, Gradient Boosting)
- Stratégies de trading algorithmique
- Développement de dashboard interactif
- Déploiement sur Hugging Face Spaces

**📊 Dataset** : Bitstamp Bitcoin Minutes Data (Kaggle)

---

## 🛠️ Technologies Utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Langage** | Python 3.x |
| **Data Science** | NumPy, Pandas |
| **Machine Learning** | Scikit-learn |
| **Visualisation** | Matplotlib, Seaborn |
| **Interface** | Streamlit |
| **Notebooks** | Jupyter |

---

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-username/Machine-learning.git
cd Machine-learning

# Créer un environnement virtuel (recommandé)
python -m venv .venv
source .venv/bin/activate  # Sur macOS/Linux
# ou
.venv\Scripts\activate     # Sur Windows

# Installer les dépendances
pip install numpy pandas matplotlib seaborn scikit-learn jupyter streamlit
```

---

## 📖 Utilisation

### Lancer les notebooks Jupyter

```bash
jupyter notebook
```

Puis naviguer vers le TP souhaité et ouvrir le fichier `.ipynb`.

### Lancer l'application Streamlit (TP2)

```bash
cd "TP-2 - use case Investor Risk Tolerance"
streamlit run app_pretty.py
```

---

## 📈 Résultats

Chaque TP contient :
- 📊 Analyses exploratoires des données
- 🤖 Modèles de machine learning entraînés
- 📉 Visualisations des performances
- 📝 Conclusions et interprétations

---

## 📄 Licence

Ce projet est réalisé dans un cadre académique à l'EMLV.

---

## 🙏 Remerciements

Nous remercions notre professeur de Machine Learning pour ses enseignements et son accompagnement tout au long de ces travaux pratiques.

---

<p align="center">
  <i>EMLV - École de Management Léonard de Vinci</i><br>
  <i>Année académique 2025-2026</i>
</p>
