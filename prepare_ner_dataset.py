import os
import json
from typing import List, Tuple

def read_annotations(annotations_dir: str) -> List[Tuple[str, List[str], List[str]]]:
    """
    Lire les annotations JSON et les convertir en format pour l'entraînement NER
    
    Returns:
    - Liste de tuples (texte, mots, labels)
    """
    dataset = []
    
    for filename in os.listdir(annotations_dir):
        if filename.endswith('_annotations.json'):
            filepath = os.path.join(annotations_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # Extraire le texte
            text = annotation.get('ocr_text', '')
            words = text.split()
            
            # Labels par défaut
            labels = ['O'] * len(words)
            
            # Appliquer les annotations
            for entity in annotation.get('annotations', []):
                entity_type = entity.get('type', '')
                entity_value = entity.get('value', '')
                
                # Trouver l'index du mot correspondant
                if entity_value in words:
                    idx = words.index(entity_value)
                    
                    # Mapper les types d'entités
                    ner_label = {
                        'Numéro de Carte': 'B-NUM_CARD',
                        'Set': 'B-SET',
                        'ID Carte': 'B-ID_CARD'
                    }.get(entity_type, 'O')
                    
                    labels[idx] = ner_label
            
            dataset.append((text, words, labels))
    
    return dataset

def save_conll_dataset(dataset: List[Tuple[str, List[str], List[str]]], output_file: str):
    """
    Sauvegarder le dataset au format CoNLL
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for text, words, labels in dataset:
            f.write(f"# {text}\n")  # Commentaire avec le texte original
            for word, label in zip(words, labels):
                f.write(f"{word} {label}\n")
            f.write("\n")  # Séparateur entre les phrases

def main():
    # Dossier contenant les annotations
    annotations_dir = os.path.join('assets', 'annotations')
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Lire les annotations
    dataset = read_annotations(annotations_dir)
    
    if not dataset:
        print("Aucune annotation trouvée. Générez d'abord des annotations avec card_annotation_ui.py")
        return
    
    # Sauvegarder au format CoNLL
    output_file = 'ner_dataset.conll'
    save_conll_dataset(dataset, output_file)
    
    print(f"Dataset NER sauvegardé dans {output_file}")
    print(f"Nombre d'exemples : {len(dataset)}")

if __name__ == "__main__":
    main()
