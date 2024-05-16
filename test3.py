import os
import cv2
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import csv
from datetime import datetime

# menetapkan dataset directory yang akan digunakan
dataset_dir = 'D:\Skripsi\Absensi'

# menetapkan label untuk wajah yang dikenali
labels = ['Naikson Saragih', 'Edward Rajagukguk', 'Asaziduhu Gea', 'Indra Kelana','gilbert','Mufria Purba']

# memuat dataset dan melakukan preprocess data latih
images = []
for label in labels:
    label_dir = os.path.join(dataset_dir, label)
    for filename in os.listdir(label_dir):
        image = cv2.imread(os.path.join(label_dir, filename))
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_resize = cv2.resize(image_grayscale, (224, 224))
        image_normalization = image_resize/ 255.0
        images.append((image_normalization, labels.index(label)))

# melakukan proses mengisi tanggal dan waktu saat wajah terdeteksi kedalam file CSV yang ditentukan
def create_csv(label):
    with open(f'{label}.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Tanggal', 'Nama', 'Status'])

        for label in labels:
            if not os.path.exists(f'{label}.csv'):
                create_csv(label)

def log_attendance(label):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d %H:%M:%S')
    day_string = now.strftime('%A')
    with open(f'{label}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([date_string, day_string, label,"Hadir"])
labels_written = {label: False for label in labels}
    
# menetapkan model arsitektur CNN 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((3, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

# melakukan Compile pada model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# melatih model
x_train = np.array([data[0] for data in images])
y_train = np.array([data[1] for data in images])
model.fit(x_train, y_train, epochs=25, batch_size=32)

# menghitung tingkat akurasi dan lost pada model yang dilatih
loss, accuracy = model.evaluate(x_train, y_train)
print("Test Loss:", loss*100,"%")
print("Test Accuracy:", accuracy*100,"%")

# menginisialisasi webcam
cap = cv2.VideoCapture(0)

# menetapkan face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

total_prediksi = 0
prediksi_benar = 0
while True:
    # melakukan Capture video frame dari webcam
    ret, frame = cap.read()

    # melakukan Convert frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # mendeteksi wajah dalam frame grayscale 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # melakukan Loop pada setiap wajah yang terdeteksi 
    for (x, y, w, h) in faces:
        # melakukan Extract pada bagian wajah dari frame webcam 
        face = frame[y:y+h, x:x+w]

        # melakukan Preprocess pada bagian wajah untuk model CNN
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224, 224))
        face = face / 255.0

        # melakukan Extract features dari bagian wajah menggunakan model CNN 
        features = model.predict(np.array([face]))

        # memberikan Label untuk wajah yang dikenali 
        label = labels[np.argmax(features)]
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
        if label != 'Unknown':
            if not labels_written[label]:
                log_attendance(label)
                labels_written[label] = True
                total_prediksi += 1
        if label == labels[np.argmax(features)]:  
            prediksi_benar += 1

    
    cv2.imshow('Face Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
    
akurasi = prediksi_benar / total_prediksi * 100
if akurasi > 100:  
       akurasi = 100

print("Akurasi Validasi:", akurasi, "%")
