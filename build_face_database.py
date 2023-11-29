import os
import cv2
import numpy as np
import tensorflow as tf

from scipy.spatial.distance import cosine
import pickle

from keras_facenet import FaceNet
# Création d'une instance FaceNet
embedder = FaceNet()

def encode_face(embedder, face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    detections = embedder.embeddings([face_rgb])
    if len(detections) > 0:
        return detections[0]
    else:
        return None


def build_database(dataset_path, embedder):
    database = {}
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        encodings = []
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                encoding = encode_face(embedder, image)  # Assurez-vous que encode_face peut accepter l'embedder en tant que paramètre
                if encoding is not None:
                    encodings.append(encoding)
        if encodings:
            database[person_name] = encodings
    return database

    
def save_database(database, filename='database.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(database, f)

def load_database(filename='database.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        return None

# Utilisation
# Construire la base de données si nécessaire
database = load_database()
if database is None:
    database = build_database('./dataset', embedder)  #
    save_database(database)
