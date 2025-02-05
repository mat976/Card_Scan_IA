import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
tf.autograph.set_verbosity(0)  # Suppress AutoGraph warnings

# Explicitly disable AutoGraph conversion for problematic functions
@tf.autograph.experimental.do_not_convert
def infer_framework(*args, **kwargs):
    # Placeholder function to prevent AutoGraph conversion warnings
    pass

import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import easyocr
import numpy as np
from transformers import (
    AutoTokenizer, 
    TFAutoModelForTokenClassification, 
    pipeline
)
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='ner_training.log',
    filemode='w'
)

class CardNERApp:
    def __init__(self, master):
        self.master = master
        master.title("Card NER Extraction Tool")
        master.geometry("1600x900")

        # Initialiser le lecteur OCR
        self.reader = easyocr.Reader(['en'])

        # Configuration du modèle
        self.model_name = "google/mobilebert-uncased"
        self.labels = ["O", "B-NUM_CARD", "B-SET", "B-ID_CARD"]
        
        # Charger le modèle et le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.labels)
        )

        # Dataset d'entraînement
        self.training_data = []

        # Frame principal
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Colonne de gauche - Images et Sélection
        self.left_frame = tk.Frame(self.main_frame, width=400)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Liste des images
        tk.Label(self.left_frame, text="Images disponibles:", font=('Arial', 12, 'bold')).pack()
        self.image_listbox = tk.Listbox(self.left_frame, width=50)
        self.image_listbox.pack(pady=5, fill=tk.BOTH, expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # Bouton Charger Images
        self.load_images_button = tk.Button(self.left_frame, text="Charger Images", command=self.load_images)
        self.load_images_button.pack(pady=5)

        # Image sélectionnée
        self.image_label = tk.Label(self.left_frame)
        self.image_label.pack(pady=10)

        # Colonne centrale - Texte et Analyse
        self.center_frame = tk.Frame(self.main_frame, width=600)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Texte Original
        tk.Label(self.center_frame, text="Texte Original:", font=('Arial', 12, 'bold')).pack()
        self.original_text = tk.Text(self.center_frame, height=10, width=70)
        self.original_text.pack(pady=5)

        # Bouton Extraire Texte
        self.extract_text_button = tk.Button(self.center_frame, text="Extraire Texte", command=self.extract_text)
        self.extract_text_button.pack(pady=5)

        # Résultats NER
        tk.Label(self.center_frame, text="Résultats NER:", font=('Arial', 12, 'bold')).pack()
        self.ner_text = tk.Text(self.center_frame, height=10, width=70)
        self.ner_text.pack(pady=5)

        # Colonne de droite - Annotations et Entraînement
        self.right_frame = tk.Frame(self.main_frame, width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Tableau des Entités
        columns = ("Type", "Valeur", "Action")
        self.ner_tree = ttk.Treeview(self.right_frame, columns=columns, show="headings")
        for col in columns:
            self.ner_tree.heading(col, text=col)
            self.ner_tree.column(col, width=120)
        self.ner_tree.pack(pady=5, expand=True, fill=tk.BOTH)
        
        # Boutons d'interaction
        button_frame = tk.Frame(self.right_frame)
        button_frame.pack(pady=5)

        self.correct_button = tk.Button(button_frame, text=" Correct", command=self.mark_correct)
        self.correct_button.pack(side=tk.LEFT, padx=5)

        self.incorrect_button = tk.Button(button_frame, text=" Incorrect", command=self.mark_incorrect)
        self.incorrect_button.pack(side=tk.LEFT, padx=5)

        self.edit_button = tk.Button(button_frame, text=" Éditer", command=self.edit_entity)
        self.edit_button.pack(side=tk.LEFT, padx=5)

        # Bouton Entraîner
        self.train_button = tk.Button(self.right_frame, text="Entraîner le Modèle", command=self.train_model)
        self.train_button.pack(pady=5)

        # État courant
        self.current_image_path = None
        self.images_list = []
        self.current_entities = []
        self.model_trained = False
        self.logger = logging.getLogger(__name__)

        # Charger les images du dossier assets par défaut
        self.load_images_from_assets()

    def load_images_from_assets(self):
        """Charger les images du dossier assets"""
        assets_dir = 'assets'
        if os.path.exists(assets_dir):
            self.images_list = [
                os.path.join(assets_dir, f) 
                for f in os.listdir(assets_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            
            # Mettre à jour la liste des images
            self.image_listbox.delete(0, tk.END)
            for img_path in self.images_list:
                self.image_listbox.insert(tk.END, os.path.basename(img_path))

    def load_images(self):
        """Charger des images depuis un dossier"""
        directory = filedialog.askdirectory(title="Sélectionner un dossier d'images")
        if directory:
            self.images_list = [
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            
            # Mettre à jour la liste des images
            self.image_listbox.delete(0, tk.END)
            for img_path in self.images_list:
                self.image_listbox.insert(tk.END, os.path.basename(img_path))

    def on_image_select(self, event):
        """Afficher l'image sélectionnée"""
        if not self.image_listbox.curselection():
            return

        index = self.image_listbox.curselection()[0]
        self.current_image_path = self.images_list[index]
        
        # Afficher l'image
        image = Image.open(self.current_image_path)
        image.thumbnail((600, 600))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

        # Réinitialiser les zones de texte
        self.original_text.delete('1.0', tk.END)
        self.ner_text.delete('1.0', tk.END)
        for i in self.ner_tree.get_children():
            self.ner_tree.delete(i)

        # Extraire automatiquement le texte
        self.extract_text()

    def extract_text(self):
        """Extraire le texte de l'image avec EasyOCR"""
        if not self.current_image_path:
            messagebox.showwarning("Attention", "Veuillez sélectionner une image.")
            return

        try:
            # Extraire le texte avec EasyOCR
            result = self.reader.readtext(self.current_image_path)
            
            # Combiner tous les textes extraits
            text = " ".join([item[1] for item in result])
            
            # Afficher le texte
            self.original_text.delete('1.0', tk.END)
            self.original_text.insert(tk.END, text)

            # Lancer automatiquement l'analyse NER
            self.analyze_ner()

        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'extraire le texte : {str(e)}")

    def analyze_ner(self):
        """Analyser le texte avec MobileBERT NER"""
        text = self.original_text.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("Attention", "Aucun texte à analyser.")
            return

        try:
            # Tokenisation manuelle
            inputs = self.tokenizer(
                text, 
                return_tensors="tf", 
                truncation=True, 
                max_length=512, 
                padding=True
            )

            # Prédiction
            outputs = self.model(inputs)
            
            # Récupérer les prédictions
            predictions = outputs.logits
            predicted_labels = tf.argmax(predictions, axis=-1)

            # Convertir en numpy pour manipulation
            input_ids = inputs['input_ids'].numpy()[0]
            label_ids = predicted_labels.numpy()[0]

            # Décoder les tokens et labels
            decoded_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # Effacer les résultats précédents
            self.ner_text.delete('1.0', tk.END)
            for i in self.ner_tree.get_children():
                self.ner_tree.delete(i)

            # Liste des entités détectées
            current_entity = None
            current_text = ""
            self.current_entities = []

            # Parcourir les tokens et labels
            for token, label_id in zip(decoded_tokens, label_ids):
                # Ignorer les tokens spéciaux
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue

                # Convertir l'ID de label en label
                label = self.labels[label_id]

                # Détecter le début d'une nouvelle entité
                if label.startswith('B-'):
                    # Ajouter l'entité précédente si existante
                    if current_entity:
                        self.current_entities.append({
                            'type': current_entity, 
                            'value': current_text, 
                            'status': 'unverified'
                        })
                        current_text = ""

                    # Commencer une nouvelle entité
                    current_entity = label[2:]
                    current_text = self.tokenizer.convert_tokens_to_string([token]).strip()
                
                # Continuer une entité existante
                elif label.startswith('I-') and current_entity:
                    current_text += " " + self.tokenizer.convert_tokens_to_string([token]).strip()
                
                # Réinitialiser si pas d'entité
                elif label == 'O':
                    if current_entity:
                        # Ajouter l'entité terminée
                        self.current_entities.append({
                            'type': current_entity, 
                            'value': current_text, 
                            'status': 'unverified'
                        })
                        current_entity = None
                        current_text = ""

                # Coloration du texte
                color_map = {
                    'NUM_CARD': 'lightblue',
                    'SET': 'lightgreen',
                    'ID_CARD': 'lightsalmon'
                }

                # Colorer si c'est une entité
                if current_entity and current_entity in color_map:
                    self.ner_text.tag_config(current_entity, background=color_map[current_entity])
                    self.ner_text.insert(tk.END, self.tokenizer.convert_tokens_to_string([token]).strip(), current_entity)
                else:
                    self.ner_text.insert(tk.END, self.tokenizer.convert_tokens_to_string([token]).strip())

            # Ajouter la dernière entité si existante
            if current_entity:
                self.current_entities.append({
                    'type': current_entity, 
                    'value': current_text, 
                    'status': 'unverified'
                })

            # Mettre à jour le tableau des entités
            for entity in self.current_entities:
                self.ner_tree.insert('', 'end', values=(entity['type'], entity['value'], 'Non vérifié'))

            # Vérifier s'il y a des résultats
            if len(self.current_entities) == 0:
                messagebox.showinfo("Résultats NER", "Aucune entité nommée n'a été détectée.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erreur NER", f"Impossible d'analyser le texte : {str(e)}")

    def prepare_training_data(self):
        """Préparer les données d'entraînement pour le modèle NER"""
        try:
            # Charger les annotations JSON
            with open('training_data/ner_annotations.json', 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            # Définir un mapping cohérent des labels
            label_map = {
                'O': 0,
                'B-NUM_CARD': 1,
                'I-NUM_CARD': 2,
                'B-SET': 3,
                'I-SET': 4,
                'B-ID_CARD': 5,
                'I-ID_CARD': 6
            }
            reverse_label_map = {v: k for k, v in label_map.items()}

            texts = []
            labels_list = []

            # Filtrer et traiter les annotations
            valid_annotations = [
                ann for ann in annotations 
                if ann.get('status') == 'correct' and ann.get('value')
            ]

            print(f"Types d'entités détectés : {set(ann['type'] for ann in valid_annotations)}")
            print(f"Nombre total d'annotations : {len(valid_annotations)}")

            for annotation in valid_annotations:
                text = annotation.get('value', '')
                entity_type = annotation.get('type', '')

                # Tokenizer avec padding et troncage
                tokenized = self.tokenizer(
                    text, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128, 
                    return_tensors='tf'
                )

                # Initialiser les labels avec 'O'
                word_labels = [label_map['O']] * len(tokenized['input_ids'][0])

                # Mapper les types d'entités aux labels
                if entity_type == 'NUM_CARD':
                    word_labels[1] = label_map['B-NUM_CARD']  # Première position après [CLS]
                elif entity_type == 'SET':
                    word_labels[1] = label_map['B-SET']
                elif entity_type == 'ID_CARD':
                    word_labels[1] = label_map['B-ID_CARD']

                texts.append(tokenized)
                labels_list.append(word_labels)

            # Vérifications supplémentaires
            print(f"Nombre de séquences de texte : {len(texts)}")
            print(f"Nombre de séquences de labels : {len(labels_list)}")
            
            # Distribution des labels
            flat_labels = np.concatenate(labels_list)
            unique_labels, label_counts = np.unique(flat_labels, return_counts=True)
            
            print("\nDistribution des labels :")
            for label, count in zip(unique_labels, label_counts):
                print(f"{reverse_label_map.get(label, 'Unknown')}: {count}")

            return texts, labels_list

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Erreur de préparation", str(e))
            return None

    def train_ner_model(self):
        """Entraîner le modèle NER avec les annotations"""
        try:
            # Préparer les données
            training_data = self.prepare_training_data()
            
            if training_data is None:
                messagebox.showerror("Erreur", "Impossible de préparer les données d'entraînement.")
                return

            texts, labels = training_data

            # Vérifier qu'il y a suffisamment de données
            if len(texts) < 10:
                messagebox.showerror("Erreur", "Pas assez de données pour l'entraînement. Ajoutez plus d'annotations.")
                return

            # Encoder les labels de manière consécutive et zéro-indexée
            unique_labels = sorted(set(np.concatenate(labels)))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            
            # Convertir les labels en indices
            encoded_labels = [[label_to_id[label] for label in seq] for seq in labels]

            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(
                texts, encoded_labels, test_size=0.2, random_state=42
            )

            # Convertir les données en format approprié avec gestion des erreurs
            def prepare_dataset(inputs, labels):
                try:
                    return {
                        'input_ids': tf.stack([x['input_ids'][0] for x in inputs]),
                        'attention_mask': tf.stack([x['attention_mask'][0] for x in inputs])
                    }, tf.ragged.constant(labels).to_tensor(shape=[None, None])
                except Exception as e:
                    print(f"Erreur de préparation du dataset : {e}")
                    return None

            train_dataset = prepare_dataset(X_train, y_train)
            test_dataset = prepare_dataset(X_test, y_test)

            if train_dataset is None or test_dataset is None:
                messagebox.showerror("Erreur", "Impossible de préparer les données d'entraînement.")
                return

            # Nombre de labels
            num_labels = len(unique_labels)

            print(f"Nombre total de labels : {num_labels}")
            print(f"Mapping des labels : {label_to_id}")

            # Configuration de l'entraînement avec des paramètres plus stables
            learning_rate = 3e-5  # Légèrement réduit
            batch_size = 4  # Réduit pour plus de stabilité
            epochs = 5  # Réduit pour éviter le surapprentissage

            # Réinitialiser le modèle avec des configurations de régularisation
            model = TFAutoModelForTokenClassification.from_pretrained(
                'google/mobilebert-uncased', 
                num_labels=num_labels,
                from_pt=True,  # Charger les poids PyTorch si nécessaire
                ignore_mismatched_sizes=True  # Ignorer les différences de taille de couche
            )

            # Compiler le modèle avec des paramètres de régularisation
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, 
                epsilon=1e-8,  # Petit epsilon pour stabilité numérique
                clipnorm=1.0  # Écrêtage du gradient
            )
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, 
                reduction=tf.keras.losses.Reduction.AUTO
            )

            model.compile(
                optimizer=optimizer, 
                loss=loss, 
                metrics=['accuracy']
            )

            # Callbacks améliorés
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                restore_best_weights=True,
                min_delta=0.001  # Amélioration minimale requise
            )

            # Entraînement avec gestion des erreurs
            print("\nDébut de l'entraînement...")
            print(f"Taille du lot : {batch_size}")
            print(f"Taux d'apprentissage : {learning_rate}")
            print(f"Nombre d'époques : {epochs}")

            try:
                history = model.fit(
                    train_dataset[0],
                    train_dataset[1],
                    validation_data=test_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping]
                )

                # Sauvegarder le modèle
                model_save_path = os.path.join('models', 'ner_model')
                os.makedirs('models', exist_ok=True)
                model.save_pretrained(model_save_path)
                self.tokenizer.save_pretrained(model_save_path)

                messagebox.showinfo("Entraînement", "Modèle entraîné avec succès!")
                self.model_trained = True

                return history

            except Exception as e:
                messagebox.showerror("Erreur d'entraînement", str(e))
                import traceback
                traceback.print_exc()
                return None

        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            import traceback
            traceback.print_exc()
            return None

    def train_model(self):
        """Wrapper pour préparer et entraîner le modèle"""
        # Charger les annotations existantes
        try:
            with open('training_data/ner_annotations.json', 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []

        # Filtrer les entités correctes et éditées
        training_entities = [
            entity for entity in self.current_entities 
            if entity['status'] in ['correct', 'edited']
        ]

        if not training_entities:
            messagebox.showwarning("Attention", "Aucune donnée d'entraînement disponible.")
            return

        # Filtres de validation
        def is_valid_entity(entity):
            # Filtres personnalisés
            value = entity['value'].strip().lower()
            
            # Filtres pour NUM_CARD
            if entity['type'] == 'NUM_CARD':
                # Doit contenir des chiffres et potentiellement des tirets
                return len(value) >= 10 and any(c.isdigit() for c in value)
            
            # Filtres pour ID_CARD
            if entity['type'] == 'ID_CARD':
                # Doit être un identifiant court et significatif
                return len(value) >= 3 and len(value) <= 10 and any(c.isdigit() for c in value)
            
            # Filtres pour SET
            if entity['type'] == 'SET':
                # Doit être un texte court et significatif
                return len(value) >= 1 and len(value) <= 10
            
            return False

        # Filtrer et nettoyer les entités
        filtered_entities = [
            entity for entity in training_entities 
            if is_valid_entity(entity)
        ]

        # Vérifier la redondance
        unique_entities = []
        seen_values = set()
        for entity in filtered_entities:
            # Éviter les doublons stricts
            if entity['value'] not in seen_values:
                unique_entities.append(entity)
                seen_values.add(entity['value'])

        # Combiner avec les données existantes
        if unique_entities:
            # Ajouter les nouvelles entités uniques
            existing_data.extend(unique_entities)

            # Limiter à un nombre maximal d'annotations
            max_annotations = 100
            if len(existing_data) > max_annotations:
                existing_data = existing_data[-max_annotations:]

            # Sauvegarder
            with open('training_data/ner_annotations.json', 'w') as f:
                json.dump(existing_data, f, indent=2)

            # Lancer l'entraînement
            self.train_ner_model()

        else:
            messagebox.showwarning(
                "Attention", 
                "Aucune nouvelle entité valide n'a été trouvée.\n"
                "Vérifiez vos annotations."
            )

    def mark_correct(self):
        """Marquer l'entité sélectionnée comme correcte"""
        selected_item = self.ner_tree.selection()
        if not selected_item:
            messagebox.showwarning("Attention", "Veuillez sélectionner une entité.")
            return

        index = self.ner_tree.index(selected_item)
        self.current_entities[index]['status'] = 'correct'
        self.ner_tree.item(selected_item, values=(
            self.current_entities[index]['type'], 
            self.current_entities[index]['value'], 
            'Correct '
        ))

    def mark_incorrect(self):
        """Marquer l'entité sélectionnée comme incorrecte"""
        selected_item = self.ner_tree.selection()
        if not selected_item:
            messagebox.showwarning("Attention", "Veuillez sélectionner une entité.")
            return

        index = self.ner_tree.index(selected_item)
        self.current_entities[index]['status'] = 'incorrect'
        self.ner_tree.item(selected_item, values=(
            self.current_entities[index]['type'], 
            self.current_entities[index]['value'], 
            'Incorrect '
        ))

    def edit_entity(self):
        """Éditer l'entité sélectionnée"""
        selected_item = self.ner_tree.selection()
        if not selected_item:
            messagebox.showwarning("Attention", "Veuillez sélectionner une entité.")
            return

        index = self.ner_tree.index(selected_item)
        
        # Boîte de dialogue pour éditer le type et la valeur
        new_type = simpledialog.askstring(
            "Éditer Type", 
            "Entrez le nouveau type d'entité :", 
            initialvalue=self.current_entities[index]['type']
        )
        
        new_value = simpledialog.askstring(
            "Éditer Valeur", 
            "Entrez la nouvelle valeur :", 
            initialvalue=self.current_entities[index]['value']
        )

        if new_type and new_value:
            self.current_entities[index]['type'] = new_type
            self.current_entities[index]['value'] = new_value
            self.current_entities[index]['status'] = 'edited'
            
            self.ner_tree.item(selected_item, values=(new_type, new_value, 'Édité '))

    def view_annotations(self):
        # Afficher les annotations pour vérification
        annotation_text = "\n".join(
            f"{entity['type']} - {entity['value']}" 
            for entity in self.training_data
        )
        messagebox.showinfo("Annotations", annotation_text)

def main():
    root = tk.Tk()
    app = CardNERApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
