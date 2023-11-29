

import cv2
import os
import shutil

# # Loading the cascades
face_cascade = cv2.CascadeClassifier('../Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../Haarcascades/haarcascade_eye.xml')


#  fonction qui detecte et renvoie image
# def detect(gray, frame):
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
#     return frame  # Return the full image with rectangles

# root_dir = './dataset'
# dirs = ['User.8', 'User.9']  # Directories to save the processed images

# # Ensure the destination directories exist
# for cls in dirs:
#     dest_dir = os.path.join(root_dir, cls)
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)

# # Process images for each class
# for cls in dirs:
#     src_dir = os.path.join(root_dir, cls)
#     files = [file for file in os.listdir(src_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     count = 1
#     for file in files:
#         src_file_path = os.path.join(src_dir, file)
#         dest_file_path = os.path.join(src_dir, f"{count}.jpg")  # Save with a new count name

#         # Read the image and convert to grayscale
#         img = cv2.imread(src_file_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Detect the face and eyes, get the image with rectangles
#         tetected_face_img = detect(gray, img)

#         # Write the processed image to the destination
#         cv2.imwrite(dest_file_path, tetected_face_img)

#         # Increment the count
#         count += 1


#  fonction qui detecte et renvoie image du visage (face ROI)
def detect(gray): 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        return roi_gray  # Return the face region only
    return None  # Return None if no faces are detected

root_dir = './dataset'
dirs = ['User.8', 'User.9']  # dossiers qui contiennent les image à prétraiter

# pour les classes dans dirs
for cls in dirs:
    src_dir = os.path.join(root_dir, cls)
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)  # 
    
    files = [file for file in os.listdir(src_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for file in files:
        src_file_path = os.path.join(src_dir, file)
        
        # lir l'image et la convertir en grayscale
        img = cv2.imread(src_file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face
        face_roi = detect(gray)

        if face_roi is not None:
            #
            dest_file_path = os.path.join(src_dir, file)  # 
            # ecraser l'image d'origine et sauvegarder face ROI
            cv2.imwrite(dest_file_path, face_roi)

