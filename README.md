# Card Scan IA - Outil d'Annotation de Cartes

## Prérequis

### 1. Python
- Python 3.10 ou supérieur recommandé

### 2. Tesseract OCR
1. **Téléchargement** :
   - Visitez : https://github.com/UB-Mannheim/tesseract/wiki
   - Téléchargez la dernière version pour Windows

2. **Installation** :
   - Exécutez l'installateur
   - **Important** : Notez le chemin d'installation (par défaut : `C:\Program Files\Tesseract-OCR`)

3. **Configuration du PATH** :
   - Ouvrez "Variables d'environnement" dans les paramètres système
   - Dans "Variables système", éditez "Path"
   - Ajoutez le chemin complet du dossier Tesseract (ex: `C:\Program Files\Tesseract-OCR`)

### 3. Dépendances Python
```bash
pip install -r requirements.txt
```

## Configuration du Projet

1. **Créez un dossier `assets`**
   ```bash
   mkdir assets
   ```

2. **Ajoutez vos images de cartes**
   - Placez vos images PNG, JPG dans le dossier `assets`

## Lancement de l'Application

### Interface d'Annotation
```bash
python card_annotation_ui.py
```

### Entraînement du Modèle NER
```bash
python train_ner_model.py
```

## Fonctionnalités

- **Interface Utilisateur** :
  - Charger des images de cartes
  - Extraction automatique de texte via OCR
  - Annotation automatique des entités
  - Sauvegarde des annotations

- **Modèle NER** :
  - Entraînement sur MobileBERT
  - Extraction d'informations spécifiques
  - Exportation en TensorFlow et TFLite

## Dépannage

- **Tesseract non reconnu** :
  - Vérifiez le chemin dans `train_ner_model.py`
  - Modifiez `pytesseract.pytesseract.tesseract_cmd` si nécessaire

- **Aucune image traitée** :
  - Assurez-vous que les images sont dans le dossier `assets/`
  - Vérifiez les formats (PNG, JPG, JPEG)

## Licence
[À compléter]

## Auteur
Demontis

## Description
Projet de reconnaissance de texte et d'entités nommées (NER) pour l'analyse de cartes.

## Installation

1. Cloner le dépôt
```bash
git clone https://github.com/votre_nom/Card_Scan_IA.git
cd Card_Scan_IA
```

2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Utilisation

### Entraînement du modèle NER

Le script `train_ner_model.py` permet d'entraîner un modèle MobileBERT pour la reconnaissance d'entités nommées :

```bash
python train_ner_model.py
```

Ce script génère un dataset synthétique et entraîne un modèle capable de reconnaître :
- Numéros de carte (`B-NUM_CARD`)
- Sets de cartes (`B-SET`)
- Identifiants de cartes (`B-ID_CARD`)

### Interface d'annotation

Utilisez l'interface graphique pour annoter des images manuellement :

```bash
python card_annotation_ui.py
```

## Fonctionnalités

- Extraction de texte avec Tesseract OCR
- Interface utilisateur Tkinter
- Modèle NER basé sur MobileBERT
- Génération de dataset synthétique

## Dépannage

### Problèmes courants

1. **Tesseract non installé** : 
   - Téléchargez et installez Tesseract OCR
   - Ajoutez le chemin d'installation au PATH système

2. **Erreurs de dépendances** :
   - Assurez-vous d'utiliser Python 3.8+
   - Vérifiez que toutes les dépendances sont installées

## Contributions

Les contributions sont les bienvenues ! Veuillez ouvrir une issue ou soumettre une pull request.

## Licence

[Spécifiez votre licence]
