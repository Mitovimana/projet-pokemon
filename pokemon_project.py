"""
PROJET POKÉMON - PRÉDICTION DE VICTOIRE
Toutes les étapes réunies dans un seul fichier
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ÉTAPE 1 : DÉFINIR LE PROBLÈME À RÉSOUDRE
# ============================================================================

def etape1_definition_probleme():
    """Définit le problème et crée les dossiers nécessaires"""
    print("\n" + "="*60)
    print("ÉTAPE 1 : DÉFINITION DU PROBLÈME")
    print("="*60)
    
    # Création des dossiers
    os.makedirs("data", exist_ok=True)
    os.makedirs("modele", exist_ok=True)
    
    print("\n🎯 Objectif : Préconiser au dresseur de Pokémon l'animal à utiliser")
    print("            lors d'un combat afin d'être le vainqueur.")
    print("\n📊 Approche :")
    print("   1. Analyser les caractéristiques des Pokémon")
    print("   2. Étudier les résultats des combats passés")
    print("   3. Créer un modèle prédictif basé sur les statistiques")
    print("   4. Prédire le pourcentage de victoire d'un Pokémon")
    print("   5. Aider le dresseur à faire le meilleur choix")


# ============================================================================
# ÉTAPE 2 : ACQUÉRIR LES DONNÉES
# ============================================================================

def etape2_acquisition_donnees():
    """Vérifie et affiche les fichiers disponibles"""
    print("\n" + "="*60)
    print("ÉTAPE 2 : ACQUISITION DES DONNÉES")
    print("="*60)
    
    fichiers = os.listdir("data")
    fichiers_csv = [f for f in fichiers if f.endswith('.csv')]
    
    print(f"\n📁 Fichiers trouvés dans data/ ({len(fichiers_csv)} fichiers) :")
    for fichier in fichiers_csv:
        taille = os.path.getsize(f"data/{fichier}")
        print(f"   - {fichier} ({taille:,} octets)")
    
    return fichiers_csv


# ============================================================================
# ÉTAPE 3 : PRÉPARATION DES DONNÉES (POKEDEX)
# ============================================================================

def etape3_preparation_pokedex():
    """Charge et prépare les données du Pokedex"""
    print("\n" + "="*60)
    print("ÉTAPE 3 : PRÉPARATION DES DONNÉES - POKEDEX")
    print("="*60)
    
    # Chargement
    print("\n📖 Chargement du Pokedex...")
    pokedex = pd.read_csv("data/pokedex.csv")
    print(f"   Shape : {pokedex.shape}")
    print(f"   Colonnes : {list(pokedex.columns)}")
    
    # Transformation de LEGENDAIRE
    print("\n🔄 Transformation de la colonne LEGENDAIRE...")
    pokedex['LEGENDAIRE'] = pokedex['LEGENDAIRE'].apply(
        lambda x: x == 'TRUE' if isinstance(x, str) else x
    )
    pokedex['LEGENDAIRE'] = pokedex['LEGENDAIRE'].astype(int)
    
    # Vérification des valeurs manquantes
    print("\n🔍 Vérification des valeurs manquantes...")
    valeurs_manquantes = pokedex.isnull().sum()
    if valeurs_manquantes.sum() > 0:
        print(valeurs_manquantes[valeurs_manquantes > 0])
    
    # Correction du nom manquant
    if pokedex['NOM'].isnull().any():
        print("\n✏️ Correction du nom manquant...")
        pokedex.loc[62, 'NOM'] = "Colossinge"
        print("   Pokémon 62 (Mankey) renommé en Colossinge")
    
    print("\n✅ Pokedex préparé avec succès!")
    return pokedex


# ============================================================================
# ÉTAPE 4 : OBSERVATION DES DONNÉES DE COMBAT
# ============================================================================

def etape4_observation_combats():
    """Analyse les données de combat"""
    print("\n" + "="*60)
    print("ÉTAPE 4 : OBSERVATION DES COMBATS")
    print("="*60)
    
    # Chargement
    print("\n⚔️ Chargement des données de combat...")
    combats = pd.read_csv("data/combats.csv")
    print(f"   Shape : {combats.shape}")
    print(f"   Colonnes : {list(combats.columns)}")
    
    # Aperçu
    print("\n📊 Aperçu des 5 premiers combats :")
    print(combats.head())
    
    # Statistiques
    print("\n📈 Statistiques descriptives :")
    print(combats.describe())
    
    # Top des victoires
    victoires = combats.groupby('GAGNANT').size().reset_index(name='VICTOIRES')
    victoires = victoires.sort_values('VICTOIRES', ascending=False)
    print("\n🏆 Top 10 des Pokémon les plus victorieux :")
    print(victoires.head(10))
    
    return combats


# ============================================================================
# ÉTAPE 5 : AGRÉGATION DES DONNÉES
# ============================================================================

def etape5_agregation(pokedex, combats):
    """Agrège les données du Pokedex et des combats"""
    print("\n" + "="*60)
    print("ÉTAPE 5 : AGRÉGATION DES DONNÉES")
    print("="*60)
    
    # Calcul des statistiques par Pokémon
    print("\n🔄 Calcul des performances par Pokémon...")
    
    # Nombre de combats
    nb_premier = combats.groupby('POKEMON_PREMIER').size().rename('NBR_PREMIER')
    nb_second = combats.groupby('POKEMON_SECOND').size().rename('NBR_SECOND')
    nb_total = nb_premier.add(nb_second, fill_value=0).rename('NBR_COMBATS')
    
    # Nombre de victoires
    nb_victoires = combats.groupby('GAGNANT').size().rename('NBR_VICTOIRES')
    
    # Fusion des données
    stats = pd.DataFrame(nb_total)
    stats = stats.join(nb_victoires, how='left')
    stats = stats.fillna(0)
    stats['POURCENTAGE'] = stats['NBR_VICTOIRES'] / stats['NBR_COMBATS']
    
    # Fusion avec le Pokedex
    dataset = pokedex.merge(stats, left_on='NUMERO', right_index=True, how='left')
    
    print(f"\n📊 Dataset final : {dataset.shape}")
    print(f"   Colonnes : {list(dataset.columns)}")
    
    # Sauvegarde
    dataset.to_csv("data/dataset.csv", index=False)
    print("\n💾 Dataset sauvegardé dans 'data/dataset.csv'")
    
    return dataset


# ============================================================================
# VISUALISATIONS
# ============================================================================

def visualisations(dataset):
    """Crée des visualisations des données"""
    print("\n" + "="*60)
    print("VISUALISATIONS")
    print("="*60)
    
    # Suppression des lignes avec valeurs manquantes pour les visualisations
    dataset_viz = dataset.dropna(subset=['POURCENTAGE'])
    
    # 1. Distribution des types
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    type1_counts = dataset_viz['TYPE_1'].value_counts()
    type1_counts.plot(kind='bar')
    plt.title('Distribution des Types Principaux')
    plt.xlabel('Type')
    plt.ylabel('Nombre de Pokémon')
    plt.xticks(rotation=90)
    
    plt.subplot(1, 2, 2)
    type2_counts = dataset_viz['TYPE_2'].value_counts()
    type2_counts.plot(kind='bar')
    plt.title('Distribution des Types Secondaires')
    plt.xlabel('Type')
    plt.ylabel('Nombre de Pokémon')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig('distribution_types.png')
    plt.show()
    
    # 2. Performance par type
    print("\n📊 Pourcentage de victoire moyen par type :")
    perf_type = dataset_viz.groupby('TYPE_1')['POURCENTAGE'].mean().sort_values()
    print(perf_type)
    
    # 3. Corrélation
    print("\n📈 Matrice de corrélation...")
    cols_corr = ['POINTS_DE_VIE', 'NIVEAU_ATTAQUE', 'NIVEAU_DEFENSE',
                 'NIVEAU_ATTAQUE_SPECIALE', 'NIVEAU_DEFENSE_SPECIALE',
                 'VITESSE', 'LEGENDAIRE', 'POURCENTAGE']
    
    corr = dataset_viz[cols_corr].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Matrice de Corrélation des Caractéristiques')
    plt.tight_layout()
    plt.savefig('matrice_correlation.png')
    plt.show()
    
    print("\n✅ Visualisations sauvegardées!")


# ============================================================================
# ÉTAPE 6 : MODÉLISATION
# ============================================================================

def etape6_modelisation(dataset):
    """Entraîne et évalue les modèles"""
    print("\n" + "="*60)
    print("ÉTAPE 6 : MODÉLISATION")
    print("="*60)
    
    # Préparation des données
    print("\n🔧 Préparation des données pour l'apprentissage...")
    dataset_clean = dataset.dropna(subset=['POURCENTAGE'])
    
    # Caractéristiques
    X = dataset_clean[['POINTS_DE_VIE', 'NIVEAU_ATTAQUE', 'NIVEAU_DEFENSE',
                       'NIVEAU_ATTAQUE_SPECIALE', 'NIVEAU_DEFENSE_SPECIALE',
                       'VITESSE', 'GENERATION', 'LEGENDAIRE']].values
    
    # Cible
    y = dataset_clean['POURCENTAGE'].values
    
    print(f"   Échantillons : {len(X)}")
    print(f"   Caractéristiques : {X.shape[1]}")
    
    # Division
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train : {len(X_train)} échantillons")
    print(f"   Test : {len(X_test)} échantillons")
    
    # Modèles
    models = {
        'Régression Linéaire': LinearRegression(),
        'Arbre de Décision': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    }
    
    results = {}
    
    for nom, modele in models.items():
        print(f"\n📊 Entraînement du modèle : {nom}")
        
        # Entraînement
        modele.fit(X_train, y_train)
        
        # Prédictions
        y_pred = modele.predict(X_test)
        
        # Évaluation
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        results[nom] = {'modele': modele, 'r2': r2, 'mse': mse}
        
        print(f"   R² : {r2:.4f}")
        print(f"   MSE : {mse:.4f}")
    
    # Meilleur modèle
    meilleur = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\n🏆 Meilleur modèle : {meilleur[0]}")
    print(f"   R² : {meilleur[1]['r2']:.4f}")
    
    # Sauvegarde
    joblib.dump(meilleur[1]['modele'], 'modele/modele_pokemon.mod')
    print("\n💾 Modèle sauvegardé dans 'modele/modele_pokemon.mod'")
    
    return meilleur[1]['modele']


# ============================================================================
# PRÉDICTION FINALE
# ============================================================================

def predire_victoire(modele, caracteristiques):
    """Prédit le pourcentage de victoire d'un Pokémon"""
    features = np.array(caracteristiques).reshape(1, -1)
    prediction = modele.predict(features)[0]
    return prediction

def tester_predictions(modele):
    """Teste le modèle avec des exemples"""
    print("\n" + "="*60)
    print("TESTS DE PRÉDICTION")
    print("="*60)
    
    exemples = [
        {
            'nom': 'Pikachu',
            'carac': [35, 55, 40, 50, 50, 90, 1, 0],
            'desc': 'Pokémon électrique classique'
        },
        {
            'nom': 'Mewtwo',
            'carac': [106, 110, 90, 154, 90, 130, 1, 1],
            'desc': 'Pokémon légendaire puissant'
        },
        {
            'nom': 'Dracaufeu',
            'carac': [78, 84, 78, 109, 85, 100, 1, 0],
            'desc': 'Pokémon de départ évolué'
        },
        {
            'nom': 'Rayquaza',
            'carac': [105, 150, 90, 150, 90, 95, 3, 1],
            'desc': 'Pokémon légendaire'
        },
        {
            'nom': 'Magikarp',
            'carac': [20, 10, 55, 15, 20, 80, 1, 0],
            'desc': 'Pokémon faible'
        }
    ]
    
    print("\n🎮 Prédictions :")
    for ex in exemples:
        pred = predire_victoire(modele, ex['carac'])
        print(f"\n   {ex['nom']} - {ex['desc']}")
        print(f"   📊 Pourcentage de victoire prédit : {pred:.2%}")
        if pred > 0.7:
            print(f"   ⭐ Recommandé pour le combat!")
        elif pred < 0.3:
            print(f"   ⚠️ À éviter pour le combat!")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale qui exécute toutes les étapes"""
    print("\n" + "🐱"*30)
    print("   PROJET POKÉMON - PRÉDICTION DE VICTOIRE")
    print("🐱"*30)
    
    # Étape 1
    etape1_definition_probleme()
    
    # Étape 2
    fichiers = etape2_acquisition_donnees()
    if not fichiers:
        print("\n❌ Erreur : Aucun fichier trouvé dans le dossier data/")
        print("   Veuillez placer les fichiers CSV dans le dossier 'data'")
        return
    
    # Étape 3
    pokedex = etape3_preparation_pokedex()
    
    # Étape 4
    combats = etape4_observation_combats()
    
    # Étape 5
    dataset = etape5_agregation(pokedex, combats)
    
    # Visualisations
    visualisations(dataset)
    
    # Étape 6
    modele = etape6_modelisation(dataset)
    
    # Tests de prédiction
    tester_predictions(modele)
    
    print("\n" + "="*60)
    print("✅ PROJET TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print("\n📁 Fichiers générés :")
    print("   - data/dataset.csv : Dataset final")
    print("   - modele/modele_pokemon.mod : Modèle entraîné")
    print("   - distribution_types.png : Graphique des types")
    print("   - matrice_correlation.png : Matrice de corrélation")


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    main()