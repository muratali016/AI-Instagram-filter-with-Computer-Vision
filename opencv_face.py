import time
import cv2
import face_recognition
import os

faces="C:/Users/murat/OneDrive/Masaüstü/Face_recog_system/faces"

def similar_face_finder(faces):
    img_database=[]
    celebrety_names=[]
    def load_images_from_folder(faces):
        for filename in os.listdir(faces):
            celebrety_names.append(filename)
            img=face_recognition.load_image_file("faces/"+filename)
            if img is not None:
                img_database.append(img)
    load_images_from_folder(faces)

    encodings_database=[]
    def encodings(img_database):
        for i in range(len(img_database)):
            img_encoder=img_database[i]
            encodingss=face_recognition.face_encodings(img_encoder)[0]
            encodings_database.append(encodingss)
    encodings(img_database)

    me=face_recognition.load_image_file("current.jpg")
    me_encode=face_recognition.face_encodings(me)[0]

    similarity_data=[]
    for i in range(len(encodings_database)):
        a=face_recognition.face_distance([me_encode], encodings_database[i])
        similarity_data.append(a)

    most_similar=min(similarity_data)
    index_of_most_similar=similarity_data.index(most_similar)
    img_rgb = cv2.cvtColor(img_database[index_of_most_similar], cv2.COLOR_BGR2RGB)
    img_rgb=cv2.resize(img_rgb,(400,350))
    celebrety_names=celebrety_names[similarity_data.index(most_similar)]
    return img_rgb,celebrety_names


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
return_value, image_currett = cap.read()
cv2.imwrite('current.jpg', image_currett)
my_similar_face,celebrety_name=similar_face_finder(faces)

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facess = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in facess:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Press T to see your celebrity", ((x-120)+(x-w),(y-120)+(y-h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2, cv2.LINE_AA)
    cv2.imshow('YOU', img)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facess = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in facess:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Celebrity you look like: ", ((x-120)+(x-w),(y-120)+(y-h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2, cv2.LINE_AA)
        cv2.putText(img, celebrety_name[:-3], ((x-80) + (x - w), (y-80) + (y - h)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('YOU', img)
    cv2.imshow("SIMILAR", my_similar_face)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

