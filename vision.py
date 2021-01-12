import cv2 as cv

DOG = 'dog.jpeg'
FACES = 'faces.jpeg'

def viewImage(image, name_of_window):
    cv.namedWindow(name_of_window, cv.WINDOW_NORMAL)
    cv.imshow(name_of_window, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#1 - Конвертирование цвета
def convertColor():
    image = cv.imread(DOG)
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    viewImage(rgb_image, "Color convertation Dog")

#2 - Кадрирование
def sampling():
    image = cv.imread(DOG)
    cropped_image = image[10:500, 500:2000] # y:y+height, x:x+width
    viewImage(cropped_image, "Cropped Dog")

#3 - Изменение размера
def scaling():
    image = cv.imread(DOG)
    scale_percent = 20
    height = int(image.shape[0] * scale_percent / 100)
    width = int(image.shape[1] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    viewImage(resized, "Resized Dog")

#4 - Поворот
def rotation():
    image = cv.imread(DOG)
    (h, w, _) = image.shape
    center = (w // 2, h // 2)
    matx = cv.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv.warpAffine(image, matx, (w, h))
    viewImage(rotated, "Rotated Dog")

#5 - Градации серого и черно-белое изображение
def grayShades():
    image = cv.imread(DOG)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, threshold_image = cv.threshold(image, 140, 255, 0)
    viewImage(gray_image, "Gray Dog")
    viewImage(threshold_image, "Threshold Dog")

#6 - Размытие/Сглаживание
def bluring():
    image = cv.imread(DOG)
    blurred = cv.GaussianBlur(image, (51, 51), 0)
    viewImage(blurred, "Blurred Dog")

#7 - Прямоугольники
def rectangles():
    image = cv.imread(DOG)
    output = image.copy()
    cv.rectangle(output, (250, 50), (900, 780), (0, 255, 255), 10)
    viewImage(output, "Rectangled Dog")

#8 - Линии
def lines():
    image = cv.imread(DOG)
    output = image.copy()
    cv.line(output, (550,200), (470,450), (0, 255, 0), 10)
    viewImage(output, "Lined Dog")

#9 - Текст
def text():
    image = cv.imread(DOG)
    output = image.copy()
    cv.putText(output, "We <3 Dogs", (80, 500), cv.FONT_HERSHEY_SIMPLEX, 5, (30, 105, 210), 20)
    viewImage(output, "Texted Dog")

#10 - Распознавание лиц
def face_detection():
    image = cv.imread(FACES)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))
    faces_detected = "Лиц обнаружено: " + format(len(faces))
    print(faces_detected)
    
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    viewImage(image, faces_detected)
    return image

#11 - Созранение изображения
def saveImage():
    image = face_detection()
    cv.imwrite('detected_faces.jpeg', image)

if __name__ == '__main__':
    saveImage()