import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine

from keras_facenet import FaceNet
# Création d'une instance FaceNet

import pickle

embedder = FaceNet()
class FaceRecognizer:
    def __init__(self):
        self.embedder = FaceNet()
        self.database_of_known_faces = self.load_database()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def load_database(self, filename='database.pkl'):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            # Si le fichier n'existe pas, construire la base de données
            # return self.build_database('./dataset')
            print ('Erreur database')
    # def build_database(self, dataset_path):
    #     database = {}
    #     for person_name in os.listdir(dataset_path):
    #         person_path = os.path.join(dataset_path, person_name)
    #         encodings = []
    #         for image_name in os.listdir(person_path):
    #             image_path = os.path.join(person_path, image_name)
    #             image = cv2.imread(image_path)
    #             if image is not None:
    #                 encoding = self.encode_face(image)
    #                 if encoding is not None:
    #                     encodings.append(encoding)
    #         if encodings:
    #             database[person_name] = encodings
    #     return database

    def encode_face(self, face):
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        detections = self.embedder.embeddings([face_rgb])
        if len(detections) > 0:
            return detections[0]
        else:
            return None

    def compare_faces(self, embedding):
        min_distance = float('inf')
        name_closest_match = None

        for (name, db_embeddings) in self.database_of_known_faces.items():
            for db_embedding in db_embeddings:
                dist = cosine(embedding, db_embedding)
                if dist < min_distance:
                    min_distance = dist
                    name_closest_match = name

        if min_distance > 0.5:  # Ajuste le seuil selon vos besoins
            return None
        return name_closest_match
    


    def detect_face(self, frame):
        # mettre l'image est en couleur (RGB) si elle ne l'est pas
        face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FaceNet s'attend à recevoir une liste d'images, même pour une seule image
        detections = self.embedder.embeddings([face_rgb])
        
        # Si detections contient des données, elle les renvoie
        if len(detections) > 0:
            return detections[0]
        else:
            return None
        #     x, y, w, h = detections[0]
        #     face_image = frame[y:y+h, x:x+w]
        #     return (x, y, w, h), face_image  # Retournez un tuple et l'image du visage
        # else:
        #     return None, None  # Retournez None si aucun visage n'est détecté
        
    def recognize_from_cam(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Erreur de capture")
                break
            detection = self.detect_face(frame)
            if detection is not None:
                name = self.compare_faces(detection)
                display_name = name if name else "Personne inconnue"
                resized_frame = cv2.resize(frame, (540, 380))
                
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = display_name
                # Définir le texte et ses propriétés

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.75
                thickness = 2

                # Calculer la taille du texte
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                # Calculer les coordonnées x et y pour centrer le texte
                text_x = (resized_frame.shape[1] - text_size[0]) // 2
                text_y = (resized_frame.shape[0] + text_size[1]) // 2  # Centrer verticalement

                # Appliquer le texte sur la frame
                cv2.putText(resized_frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

 
            # face_location, face_image = self.detect_face(frame)
            # if face_location is not None:
                
            #     x, y, w, h = face_location  # Extraction des coordonnées et dimensions du visage
            #     encoding = self.encode_face(face_image)
            #     name = self.compare_faces(encoding)
            #     display_name = name if name else "Personne inconnue"
                
            #     # Code pour afficher le nom et le rectangle
            #     cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            #     # ... Code pour afficher la frame et quitter avec 'q' ...
            #     cv2.imshow('Video', frame)
                # Quitter le flux vidéo avec 'q'
            else:
                print("Aucun visage détecté dans l'image.")
        
            cv2.imshow('Image', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

    def recognize_image(self, image_path):
        
        frame = cv2.imread(image_path)
        if frame is not None:
            # face_location, face_image = self.detect_face(frame)
            # if face_location is not None:
            #     x, y, w, h = face_location
            #     encoding = self.encode_face(face_image)
            detection = self.detect_face(frame)
            if detection is not None:
                name = self.compare_faces(detection)
                display_name = name if name else "Personne inconnue"
                resized_frame = cv2.resize(frame, (540, 380))
                
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = display_name
                # Définir le texte et ses propriétés

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3

                # Calculer la taille du texte
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                # Calculer les coordonnées x et y pour centrer le texte
                text_x = (resized_frame.shape[1] - text_size[0]) // 2
                text_y = (resized_frame.shape[0] + text_size[1]) // 2  # Centrer verticalement

                # Appliquer le texte sur la frame
                cv2.putText(resized_frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)



                # font_scale = 0.75
                # font = cv2.FONT_HERSHEY_SIMPLEX

                # # Calculer la taille du texte
                # (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 2)

                # # Calculer x (pour centrer le texte) et y (position fixe en haut)
                # text_x = (frame.shape[1] - text_width) // 2  # frame.shape[1] est la largeur de l'image
                # text_y = 30  # ou toute autre valeur fixe en haut

                # # Appliquer le texte sur l'image redimensionnée
                # cv2.putText(resized_frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), 2)

                # Afficher l'image redimensionnée
                cv2.imshow('Image', resized_frame)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Aucun visage détecté dans l'image.")
        else:
            print("Image non trouvée ou chemin invalide.")

# Utilisation
# face_recognizer = FaceRecognizer()
# face_recognizer.recognize_from_cam()
# face_recognizer.recognize_image('chemin/vers/image.jpg')