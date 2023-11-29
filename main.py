
from  script_face_recognise_cam  import FaceRecognizer

# Créer une instance de la classe
face_recognizer = FaceRecognizer()
 
path_img_hous = './dataset/Houssine/6.2.jpg'
path_img_hous2 = './dataset/Houssine/7.22.jpg'
path_img_Sof = './dataset/Sofiane/1.jpg'
path_img_Athman = './dataset/Athman/3.5.jpg'
path_img_Rachida = './dataset/Rachida/1.5.jpg'
path_img_Thibaut = './dataset/Thibaut/9.jpg'
path_img_khalil = './dataset/Khalil/10.2.jpg'
# image complete couleur:
path_img_Sof2 = './test_images/IMG_20231124_184715.jpg'
path_img_Sof3 = './test_images/IMG_20231124_185017.jpg'
path_img_Thibaut2 = './test_images/IMG_20231125_122744.jpg'
path_img_Thibaut3 = './test_images/IMG_20231125_122820.jpg'
path_img_inconnu1 = '../celebrities_images/Aamir Khan/1.jpg'
path_img_inconnu2 = '../celebrities_images/Brad Pitt/001_c04300ef.jpg'
path_img_inconnu3 = '../celebrities_images/Sandra Bullock/005_b0b4e2fa.jpg'
path_img_inconnu4 = '../celebrities_images/Angelina Jolie/Image_67.jpg'
path_img_inconnu5 = '../celebrities_images/Vin Diesel/Image_54.jpg'
# Appeler la méthode avec le chemin de l'image
#face_recognizer.recognize_image(path_img_khalil)
#face_recognizer.recognize_image(path_img_hous2)
#face_recognizer.recognize_image(path_img_inconnu5)

# test avec camera:
face_recognizer.recognize_from_cam()