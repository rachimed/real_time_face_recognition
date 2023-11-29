''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition     

# '''

import cv2
import os
import time
# Loading the cascades
face_cascade = cv2.CascadeClassifier('../Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../Haarcascades/haarcascade_eye.xml')


# Vérifie l'existence du dossier 'dataset' et crée s'il n'existe pas
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Trouve l'ID maximum existant dans le dossier 'dataset' et incrémente de 1 pour la nouvelle personne

person_ids = []

# Parcourt tous les dossiers et fichiers dans le dossier 'dataset'
for item in os.scandir('dataset'):
    # Vérifie si l'entrée est un dossier
    if item.is_dir():
        # Extrait le nom du dossier
        folder_name = item.name
        # Vérifie si le nom du dossier commence par 'User.'
        if folder_name.startswith('User.'):
            # Extrait l'ID de l'utilisateur du nom du dossier, qui suit 'User.'
            person_id = folder_name.replace('User.', '')
            # Convertit l'ID de l'utilisateur en un entier et l'ajoute à la liste des IDs
            person_ids.append(int(person_id))

# Après avoir collecté tous les identifiants, 'person_ids' contient tous les IDs numériques des dossiers existants.

face_id = max(person_ids) + 1 if person_ids else 1

# Crée un sous-dossier pour la nouvelle personne dans 'dataset'
person_path = os.path.join('dataset', f'User.{face_id}')
os.makedirs(person_path, exist_ok=True)

print(f"\n [INFO] Initializing face capture for user ID {face_id}. Look the camera and wait ...")
count = 0

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Initialise la capture vidéo
cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set video width
# cam.set(4, 480)  # set video height

while(True):

    ret, img = cam.read()
    # if not ret:
    #     break
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
   
    # canvas = detect(gray, img)
        count += 1
        cv2.imwrite(f"{person_path}/{face_id}.{count}.jpg", roi_gray)
       
        cv2.imshow('Video',img )      
        time.sleep(1)
    # k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    # if k == 27:
    #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 30:  # Prend 30 échantillons de visage et arrête la vidéo
        break

# Nettoyage
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
