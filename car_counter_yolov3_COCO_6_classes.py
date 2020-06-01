
'''
Программа распознает 6 типов автомобилей на видео, производит подсчет каждого из них, а также 
подсчет общего числа атомобилей на данной кадре и заносит все резльтаты в .json файл
'''

# RUN THE CODE:
# python car_counter_yolov3_6_classes.py -y yolo --input videos/traffic.mp4 --output output --skip-frames 5

# импортируем необходимые библиотеки и функции
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import dlib
import json
import cv2
import os
from matplotlib import pyplot as plt

# парсер аргументов с командной строки
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required = True, type=str,
	help = "path to yolo directory")
ap.add_argument("-i", "--input", required = True, type=str,
	help="path to input video file")
ap.add_argument("-o", "--output", required = True, type=str,
	help="path to output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.90,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="number of frames to skip between detections"
		 "the higher the number the faster the program works")
args = vars(ap.parse_args())


# Функция считает общее количество объектов класса, появившихся на видео
def count_objects(objects, object_class, total, temp):
	global total_cars
	global temp_cars
	global total_persons
	global temp_persons
	global total_trucks
	global temp_trucks
	global total_buses
	global temp_buses
	global total_bikes
	global temp_bikes
	global total_bicycles
	global temp_bicycles


	if object_class == "car":
		total, temp = total_cars, temp_cars
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		# переприсваиваем глобальную переменную
		total_cars = total
		temp_cars = temp	

	elif object_class == "person":
		total, temp = total_persons, temp_persons
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		# переприсваиваем глобальную переменную
		total_persons = total
		temp_persons = temp	

	elif object_class == "truck":
		total, temp = total_trucks, temp_trucks
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		# переприсваиваем глобальную переменную
		total_trucks = total
		temp_trucks = temp	
	elif object_class == "bus":

		total, temp = total_buses, temp_buses
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		# переприсваиваем глобальную переменную
		total_buses = total
		temp_buses = temp	
	elif object_class == "bike":
		total, temp = total_bikes, temp_bikes	
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		# переприсваиваем глобальную переменную
		total_bikes = total
		temp_bikes = temp	
	elif object_class == "bicycle":
		total, temp = total_bicycles, temp_bicycles		
		length = len(objects.keys())
		if length > total:
			total += length - total
		if temp is not None:
			if (length > temp):
				total += length - temp
		if length < total:
			temp = length
		# переприсваиваем глобальную переменную
		total_bicycles = total
		temp_bicycles = temp	
	
	
	# возвращаем количество авто одного типа
	return total	



# Функция рисует ID-шники и центроиды объектов
def draw_centroids(frame, objects, trackableObjects):
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




# Настройка yolov3 СТОК
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["yolo"] + "/yolov3_608.weights", args["yolo"] + "/yolov3_608.cfg")
print("[INFO] path to weights: ", args["yolo"] + "/yolov3_608.weights")
print("[INFO] path to cfg: ", args["yolo"] + "/yolov3_608.cfg")
# классы объектов, которые могут быть распознаны алгоритмом
with open(args["yolo"] + "/yolov3_608.names", 'r') as f:
	CLASSES = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Размеры входного изображения
inpWidth = 608
inpHeight = 608

# путь к исходному видео
print("[INFO] input directory: ", args["input"])

# читаем видео с диска
print("[INFO] opening video file...")
vs = cv2.VideoCapture(args["input"])

# объявляем инструмент для записи конечного видео в файл, указываем путь
writer = None
output_count = 1
while True:
	# если в директории вывода уже больше 20 файлов, то она очищается
	if output_count > 20:
		for file in os.listdir(args["output"]):
			os.remove(os.getcwd() + "/output/" + file)
			output_count = 1
	if "{}_proccesed.avi".format(output_count) not in os.listdir(args["output"]):
		writer_path = args["output"] + "/{}_proccesed.avi".format(output_count)
		break
	else:
		output_count += 1
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
car_ct = CentroidTracker()
car_ct.maxDisappeared = 10
person_ct = CentroidTracker()
person_ct.maxDisappeared = 10
truck_ct = CentroidTracker()
truck_ct.maxDisappeared = 10
bike_ct = CentroidTracker()
bike_ct.maxDisappeared = 10
bicycle_ct = CentroidTracker()
bicycle_ct.maxDisappeared = 10
bus_ct = CentroidTracker()
bus_ct.maxDisappeared = 10

# сам список трекеров
trackers = []
# список объектов для трекинга
car_trackableObjects = {}
person_trackableObjects = {}
truck_trackableObjects = {}
bus_trackableObjects = {}
bike_trackableObjects = {}
bicycle_trackableObjects = {}

# глобальные переменные для счетчиков
total_cars, temp_cars = 0, None
total_persons, temp_persons = 0, None
total_trucks, temp_trucks = 0, None
total_buses, temp_buses = 0, None
total_bikes, temp_bikes = 0, None
total_bicycles, temp_bicycles = 0, None


# полное число кадров в видео
totalFrames = 0

total = 0

# статус: распознавание или отслеживание
status = None

#номер кадра видео
frame_number = 0

# инициализируем нулями
count_cars, count_persons, count_trucks, count_buses, count_bikes, count_bicycles = 0, 0, 0, 0, 0, 0

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

	# этот список боксов может быть заполнен двумя способами:
	# (1) детектором объектов
	# (2) трекером наложений из библиотеки dlib
	#2 автомобили
	#0 человек
	#7 грузовки
	#5 автобус
	#3 мото
	#1 велосипед
	car_rects = []
	person_rects = []
	truck_rects = []
	bus_rects = []
	bike_rects = []
	bicycle_rects = []

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
		# сколь машин на данном кадре
		count = 0

		status = "Detecting..."

		# получаем blob-модель из кадра и пропускаем ее через сеть, чтобы получить боксы распознанных объектов
		blob = cv2.dnn.blobFromImage(frame, 0.00392, (inpWidth, inpHeight), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)


		# анализируем список боксов
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				

				if class_id == 0: # если обнаружена "background" - пропускаем
					pass


				confidence = scores[class_id]
				# получаем ID наиболее "вероятных" объектов
				if confidence > args["confidence"]:
					# находятся координаты центроида бокса
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

					# возьмем максимальный радиус для CentroidTracker пропорционально размеру машины
					person_ct.maxDistance = w
					bike_ct.maxDistance = w
					bicycle_ct.maxDistance = w
					bus_ct.maxDistance = w
					truck_ct.maxDistance = w
					car_ct.maxDistance = w


					count += 1

					# рисую бокс для теста
					cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255 , 0), 1)
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
			cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255 , 0), 1)
			cv2.putText(frame, CLASSES[class_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			obj_class = CLASSES[class_id]
			#2 автомобили
			#0 человек
			#7 грузовки
			#5 автобус
			#3 мото
			#1 велосипед

			if obj_class == "car":
				car_rects.append((x1, y1, x2, y2))
			elif obj_class == "person":
				person_rects.append((x1, y1, x2, y2))	
			elif obj_class == "truck":
				truck_rects.append((x1, y1, x2, y2))	
			elif obj_class == "bus":
				bus_rects.append((x1, y1, x2, y2))			
			elif obj_class == "motorcycle":
				bike_rects.append((x1, y1, x2, y2))		
			elif obj_class == "bicycle":
				bicycle_rects.append((x1, y1, x2, y2))	

	'''
	После детекта первой машины и до конца работы программы rects больше никогда не станут []. 
	Единственное условие, при котором len(objects.keys()) станет равно 0. Это если истичет предел maxDisappeared, то есть
	rects так и будут НЕпустым массивом, но машина слишком надолго исчезнет из виду.
	'''
	cars = car_ct.update(car_rects)
	persons = person_ct.update(person_rects)
	trucks = truck_ct.update(truck_rects)
	buses = bus_ct.update(bus_rects)
	bikes = bike_ct.update(bike_rects)
	bicycles = bicycle_ct.update(bicycle_rects)



	if cars != {}:
		count_cars = count_objects(cars, "car", total_cars, temp_cars)
	if persons != {}:	
		count_persons = count_objects(persons, "person", total_persons, temp_persons)
	if trucks != {}:
		count_trucks = count_objects(trucks, "truck", total_trucks, temp_trucks)
	if buses != {}:	
		count_buses = count_objects(buses, "bus", total_buses, temp_buses)
	if bikes != {}:
		count_bikes = count_objects(bikes, "bike", total_bikes, temp_bikes)
	if bicycles != {}:
		count_bicycles = count_objects(bicycles, "bicycle", total_bicycles, temp_bicycles)


	draw_centroids(frame, cars, car_trackableObjects)
	draw_centroids(frame, persons, person_trackableObjects)
	draw_centroids(frame, trucks, truck_trackableObjects)
	draw_centroids(frame, buses, bus_trackableObjects)
	draw_centroids(frame, bikes, bike_trackableObjects)
	draw_centroids(frame, bicycles, bicycle_trackableObjects)


	# Данные для вывода на экран
	info = [
		("cars: ", count_cars),
		("people: ", count_persons),
		("trucks: ", count_trucks),
		("buses: ", count_buses),
		("bikes: ", count_bikes),
		("bicycles", count_bicycles),
	]

	# данные для записи в JSON
	data = [{
		"cars" : str(count_cars),
		"people" : str(count_persons),
		"trucks" : str(count_trucks),
		"buses:" : str(count_buses),
		"motorcycles:": str(count_bikes),
		"bycicles" : str(count_bicycles),
	}]

	# изобразим информаци о количестве машин на краю кадра
	for (i, (object_class, total)) in enumerate(info):
		text = "{}: {}".format(object_class, total)
		cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
		cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)

	cv2.putText(frame, "Now: " + str(count), (width - 120, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)	

	# записываем конечный кадр в указанную директорию
	if writer is not None:
		writer.write(frame)



	# показываем конечный кадр в отдельном окне
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# для прекращения работы необходимо нажать клавишу "q"
	if key == ord("q"):
		print("[INFO] process finished by user")

		break

	# т.к. все выше-обработка одного кадра, то теперь необходимо увеличить количесвто кадров
	# и обновить счетчик
	totalFrames += 1

# график выводится на экран в конце работы программы
plt.show()


# записываю все полученные данные в json файл
with open(args["output"] + "/" + "analysis_results_{}.json".format(output_count), 'w') as f:
	json.dump(data, f)

print("\nThe results are:")
with open(args["output"] + "/" + "analysis_results_{}.json".format(output_count), 'r') as f:
	data = json.load(f)
	for el in data:
		for key, value in el.items():
			print(key + " " + value)

# освобождаем память под переменную
if writer is not None:
	writer.release()

# закрываем все окна
cv2.destroyAllWindows()
