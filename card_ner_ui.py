import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import easyocr
import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer, 
    TFAutoModelForTokenClassification, 
    pipeline
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

    def train_model(self):
        """Entraîner le modèle avec les données annotées"""
        # Filtrer les entités correctes et éditées
        training_entities = [
            entity for entity in self.current_entities 
            if entity['status'] in ['correct', 'edited']
        ]

        if not training_entities:
            messagebox.showwarning("Attention", "Aucune donnée d'entraînement disponible.")
            return

        # Ajouter au dataset d'entraînement
        self.training_data.extend(training_entities)

        # Sauvegarder les données d'entraînement
        os.makedirs('training_data', exist_ok=True)
        with open('training_data/ner_annotations.json', 'w') as f:
            json.dump(self.training_data, f, indent=2)

        messagebox.showinfo("Entraînement", f"{len(training_entities)} entités ajoutées au dataset d'entraînement.")

def main():
    root = tk.Tk()
    app = CardNERApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
