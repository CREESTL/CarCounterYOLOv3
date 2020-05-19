
'''
Программа распознает и считает автомобили на видео с помощью YOLOv3
'''

# Запуск программы с командной строки
# cd C:\CREESTL\Programming\PythonCoding\semestr_4\CarCounterYOLOv3
# python car_counter_yolov3.py -y yolo --input videos/10fps.mp4 --output output --skip-frames 5

# импортируем необходимые библиотеки и функции
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt

# парсер аргументов с командной строки
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required = True, type=str,
	help = "path to yolo directory")
ap.add_argument("-i", "--input", required = True, type=str,
	help="path to input video file")
ap.add_argument("-o", "--output", required = True, type=str,
	help="path to output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.01,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="number of frames to skip between detections"
		 "the higher the number the faster the program works")
args = vars(ap.parse_args())



# классы объектов, которые могут быть распознаны алгоритмом
with open(args["yolo"] + "/classes.names", 'r') as f:
	CLASSES = [line.strip() for line in f.readlines()]


# Настройка yolov3
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["yolo"] + "/yolo-obj_9000.weights", args["yolo"] + "/yolo-obj.cfg")
print("path to weights: ", args["yolo"] + "/yolo-obj_9000.weights")
print("path to cfg: ", args["yolo"] + "/yolo-obj.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# путь к исходному видео
print("[INFO] input directory: ", args["input"])

# читаем видео с диска
print("[INFO] opening video file...")
vs = cv2.VideoCapture(args["input"])

# объявляем инструмент для записи конечного видео в файл, указываем путь
writer = None
writer_path = args["output"] + "\last.avi"
print("[INFO] output directory: ", writer_path)

# инициализируем размеры кадра как пустые значения
# они будут переназначены при анализе первого кадра и только
# это ускорит работу программы
width = None
height = None

# инициализируем алгоритм трекинга
# maxDisappeared = кол-во кадров, на которое объект может исчезнуть с видео и потом опять
# будет распознан
# maxDistance = максимальное расстояние между центрами окружностей, вписанных в боксы машин
# Если расстояние меньше заданного, то происходит переприсваение ID
ct = CentroidTracker()

# сам список трекеров
trackers = []
# список объектов для трекинга
trackableObjects = {}



# полное число кадров в видео
totalFrames = 0

# счетчик машин и временная переменная
total = 0
temp = None

# статус: распознавание или отслеживание
status = None


#номер кадра видео
frame_number = 0



# проходим через каждый кадр видео
while True:
	frame_number += 1
	frame = vs.read()
	frame = frame[1]


	# если кадр является пустым значением, значит был достигнут конец видео
	if frame is None:
		print("=============================================")
		print("The end of the video reached")
		print("Total number of cars on the video is ", total)
		print("=============================================")
		break

	# изменим размер кадра для ускорения работы
	frame = imutils.resize(frame, width=800)

	# для работы библиотеки dlib необходимо изменить цвета на RGB вместо BGR
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# размеры кадра
	if width is None or height is None:
		height, width, channels = frame.shape

	# в зависимости от размера кадра настраиваем минимальный радиус и максимальное число кадром длс centroid tracker
	ct.maxDistance = width / 2
	ct.maxDisappeared = 10

	# этот список боксов может быть заполнен двумя способами:
	# (1) детектором объектов
	# (2) трекером наложений из библиотеки dlib
	rects = []

	# задаем путь записи конечного видео
	if  writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(writer_path,fourcc, 30,
			(width, height), True)



	# каждые N кадров (указанных в аргументе "skip_frames" производится ДЕТЕКТРОВАНИЕ машин
	# после этого идет ОТСЛЕЖИВАНИЕ их боксов
	# это увеличивает скорость работы программы
	if totalFrames % args["skip_frames"] == 0:
		# создаем пустой список трекеров
		trackers = []
		# список номером классов (нужен для подписи класса у боксов машин
		class_ids = []

		status = "Detecting..."

		# получаем blob-модель из кадра и пропускаем ее через сеть, чтобы получить боксы распознанных объектов
		blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)

		# анализируем список боксов
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				# получаем ID наиболее "вероятных" объектов
				if confidence > args["confidence"]:
					print(f"CAR FOUND!")
					print(f"class id = {class_id}")

					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					# это ИМЕННО ШИРИНА - то есть расстояние от левого края до правого
					w = int(detection[2] * width)
					# это ИМЕННО ВЫСОТА - то есть расстояние от верхнего края до нижнего
					h = int(detection[3] * height)

					# Координаты бокса (2 точки углов)
					x1 = int(center_x - w / 2)
					y1 = int(center_y - h / 2)
					x2 = x1 + w
					y2 = y1 + h

					print(f"x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")
					# рисую бокс для теста
					cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
					cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

					# создаем трекер ДЛЯ КАЖДОЙ МАШИНЫ
					tracker = dlib.correlation_tracker()
					# создаем прямоугольник из бокса (фактически, это и есть бокс)
					rect = dlib.rectangle(x1, y1, x2, y2)
					# трекер начинает отслеживание КАЖДОГО БОКСА
					tracker.start_track(rgb, rect)
					# и каждый трекер помещается в общий массив
					trackers.append(tracker)
					class_ids.append(class_id)

	# если же кадр не явялется N-ым, то необходимо работать с массивом сформированных ранее трекеров, а не боксов
	else:
		for tracker, class_id in zip(trackers, class_ids):

			status = "Tracking..."

			'''
			На одном кадре машина была распознана. Были получены координаты ее бокса. ВСЕ последующие 5 кадров эти координаты
			не обращаются в нули, а изменяются благодяра update(). И каждый их этих пяти кадров в rects помещается предсказанное
			программой местоположение бокса!
			'''
			tracker.update(rgb)
			# получаем позицию трекера в списке(это 4 координаты)
			pos = tracker.get_position()

			# из трекера получаем координаты бокса, соответствующие ему
			x1 = int(pos.left())
			y1 = int(pos.top())
			x2 = int(pos.right())
			y2 = int(pos.bottom())

			# рисую бокс
			cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
			cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			# и эти координаты помещаем в главный список коодинат боксов ДЛЯ КАДРА (по нему и будет производиться рисование)
			rects.append((x1, y1, x2, y2))

	'''
	После детекта первой машины и до конца работы программы rects больше никогда не станут []. 
	Единственное условие, при котором len(objects.keys()) станет равно 0. Это если истичет предео maxDisappeared, то есть
	rects так и будут НЕпустым массивом, но машина слишком надолго исчезнет из виду.
	'''
	print(f"rects = {rects}")
	objects = ct.update(rects)


	# алгоритм подсчета машин
	length = len(objects.keys())
	print(f"objects length = {length}")
	if length > total:
		print(f"length > total")
		total += length - total
	if temp is not None:
		if (length > temp):
			print("length > temp")
			total += length - temp
	if length < total:
		print(f"length < total")
		temp = length
	print(f"total is {total}")
	print(f"temp is {temp}\n")

	# анализируем массив отслеживаемых объектов
	for (objectID, centroid) in objects.items():
		# проверяем существует ли отслеживаемый объект для данного ID
		to = trackableObjects.get(objectID, None)

		# если его нет, то создаем новый, соответствующий данному центроиду
		if to is None:
			to = TrackableObject(objectID, centroid)

		# в любом случае помещаем объект в словарь
		# (1) ID (2) объект
		trackableObjects[objectID] = to


		# изобразим центроид и ID объекта на кадре
		text = "ID {}".format(objectID + 1)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	info = [
		("Total", total),
		("Status", status)
	]

	# изобразим информаци о количестве машин на краю кадра
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
		cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)

	# записываем конечный кадр в указанную директорию
	if writer is not None:
		writer.write(frame)



	# показываем конечный кадр в отдельном окне
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# для прекращения работы необходимо нажать клавишу "q"
	if key == ord("q"):
		print("[INFO] process finished by user")
		print("Total number of cars on the video is ", total)
		break

	# т.к. все выше-обработка одного кадра, то теперь необходимо увеличить количесвто кадров
	# и обновить счетчик
	totalFrames += 1

# график выводится на экран в конце работы программы
plt.show()

# освобождаем память под переменную
if writer is not None:
	writer.release()

# закрываем все окна
cv2.destroyAllWindows()
