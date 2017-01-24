import os
import xml.etree.ElementTree as ET
import numpy as np 
import pickle
import config as cfg
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class data(object):

	def __init__(self):
		self.path = 'data/Pickle/train_4_22279.pkl'
		# self.path = 'data/Pickle/train55.pkl'
		self.epoch = 1
		self.batch_size = cfg.BATCH_SIZE
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.classes = cfg.CLASSES
		self.cursor = 0
		self.epoch = 1

	def load(self):
		with open(self.path, 'rb') as handle:
			packet = pickle.load(handle)
		self.X_train = packet['xtrain']
		self.y_train = packet['ytrain']
		self.X_val = packet['xvalid']
		self.y_cal = packet['yvalid']
		self.X_test = packet['xtest']
		self.y_test = packet['ytest']

	def get(self):
		# images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
		# labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25))
		count = 0
		# self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = 0)
		while count < self.batch_size:
			images_name = self.X_train[self.cursor:self.cursor+self.batch_size]
			images = self.getimages(images_name)
			# images = self.X_train[self.cursor:self.cursor+self.batch_size]
			labels = self.y_train[self.cursor:self.cursor+self.batch_size]

			# print (labels.shape)
			self.cursor += self.batch_size
			if self.cursor+self.batch_size >= self.X_train.shape[0]:
				self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = 0)
				self.cursor = 0
				self.epoch += 1
			return images, labels

	def getimages(self, names):
		images = []
		for fname in names:
			img = cv2.imread(fname)
			img = cv2.resize(img, (self.image_size, self.image_size))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = (img / 255.0) * 2.0 - 1.0
			images += [img]
		images = np.array(images).astype(np.float32)
		return images


class voc(object):

	def __init__(self):
		self.devkil_path = os.path.join('data', 'VOCdevkit')
		self.data_path = os.path.join(self.devkil_path, 'VOC2012')
		self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
						'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
						'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
						'train', 'tvmonitor']
		self.classinterest = [1, 5, 6, 13, 14, 18]
		# self.classinterest = list(range(20))
		# ['car', 'bicycle', 'bus', 'motorbike', 'person']
		self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE

	def load_pascal(self):
		images = []
		labels = []
		txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
		with open(txtname, 'r') as f:
			self.image_index = [x.strip() for x in f.readlines()]

		for n in range(len(self.image_index)):
		# for index in self.image_index:
			index = self.image_index[n]
			img, label, num = self.load_pascal_annotation(index)
			if num != 0:
				images += [img]
				labels += [label]
			if n % 100 == 0:
				print (n, '/' , len(self.image_index), end = '\r')
		images = np.array(images)
		labels = np.array(labels).astype(np.float32)

		# X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, random_state = 42)
		# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)

		# pickle_packet = {'xtrain' : X_train,
		# 				 'ytrain' : y_train,
		# 				 'xvalid' : X_val,
		# 				 'yvalid' : y_val,
		# 				 'xtest'  : X_test,
		# 				 'ytest'  : y_test}
		# pickle_name = 'data/Pickle/' + 'voctrain' + str(images.shape[0]) + '.pkl'

		# # if os.path.exists(pickle_name):
		# with open(pickle_name, 'wb') as f:
		# 	pickle.dump(pickle_packet, f)
		# 	print (pickle_name + ' saved')
		return images, labels


	def load_pascal_annotation(self, index):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC
		format.
		"""
		imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
		im = cv2.imread(imname)
		h_ratio = 1.0 * self.image_size / im.shape[0]
		w_ratio = 1.0 * self.image_size / im.shape[1]
		# im = cv2.resize(im, [self.image_size, self.image_size])

		label = np.zeros((self.cell_size, self.cell_size, 5+len(self.classes)))
		filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
		tree = ET.parse(filename)
		objs = tree.findall('object')

		# img = cv2.resize(im, (self.image_size, self.image_size))
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# img = (img / 255.0) * 2.0 - 1.0

		nn = 0
		for obj in objs:
			bbox = obj.find('bndbox')
			# Make pixel indexes 0-based
			x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
			y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
			x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
			y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
			cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
			if cls_ind in self.classinterest:
				nn += 1
				boxes = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]
				x_ind = int(boxes[0] * self.cell_size / self.image_size)
				y_ind = int(boxes[1] * self.cell_size / self.image_size)
				if label[y_ind, x_ind, 0] == 1:
					continue
				label[y_ind, x_ind, 0] = 1
				label[y_ind, x_ind, 1:5] = boxes
				label[y_ind, x_ind, 5 + cls_ind] = 1
		return imname, label, nn


class crowdai(object):
	def __init__(self):
		self.datapath = 'data/object-detection-crowdai/'
		self.labelpath = 'data/object-detection-crowdai/labels.csv'
		self.classes = ['Truck', 'Car', 'Pedestrian']
		# self.classinterest = [0, 6, 14]
		self.classinterest = [0,1,2]
		# self.classinterest = list(range(20))
		# ['car', 'bicycle', 'bus', 'motorbike', 'person']
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE

	def load_crowdai(self):
		import csv
		imname = None
		images = []
		labels = []
		with open(self.labelpath, newline='') as f:
			reader = csv.reader(f)
			r = 0
			ri = 0
			for row in reader:
				r += 1
				if imname is None:
					imname = ''
					pass
				else:
					if imname == '':
						imname = self.datapath + row[4]
						img = cv2.imread(imname)
						h_img = img.shape[0]
						w_img = img.shape[1]
						h_ratio = 1.0 * self.image_size / img.shape[0]
						w_ratio = 1.0 * self.image_size / img.shape[1]
						label = np.zeros((self.cell_size, self.cell_size, 5+20))
					else:
						if self.datapath + row[4] == imname:
							pass
						else:
							images += [imname]
							labels += [label]
							ri += 1
							print (imname, label.shape, r,'/72065', ri, end='\r')
							imname = self.datapath + row[4]
							img = cv2.imread(imname)
							label = np.zeros((self.cell_size, self.cell_size, 5+20))
					# x1 = int(row[0])
					# x2 = int(row[2])
					# y1 = int(row[1])
					# y2 = int(row[3])

					x1 = (float(row[0]) - (w_img/2)) * w_ratio + (224)
					y1 = (float(row[1]) - (h_img/2)) * h_ratio + (224)
					x2 = (float(row[2]) - (w_img/2)) * w_ratio + (224)
					y2 = (float(row[3]) - (h_img/2)) * h_ratio + (224)
					class_name = row[5]
					for i in range(3):
						if class_name == self.classes[i]:
							cls_ind = self.classinterest[i]
					# x1 = max(min((float(row[0]) - 1 - (w_img/2)) * w_ratio + (w_img/2), self.image_size - 1), 0)
					# y1 = max(min((float(row[1]) - 1 - (h_img/2)) * h_ratio + (h_img/2), self.image_size - 1), 0)
					# x2 = max(min((float(row[2]) - 1 - (w_img/2)) * w_ratio + (w_img/2), self.image_size - 1), 0)
					# y2 = max(min((float(row[3]) - 1 - (h_img/2)) * h_ratio + (h_img/2), self.image_size - 1), 0)
					boxes = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]
					x_ind = int(boxes[0] * self.cell_size / self.image_size)
					y_ind = int(boxes[1] * self.cell_size / self.image_size)
					if label[y_ind, x_ind, 0] == 1:
						continue
					label[y_ind, x_ind, 0] = 1
					label[y_ind, x_ind, 1:5] = boxes
					label[y_ind, x_ind, 5 + cls_ind] = 1
			
		images = np.array(images)
		labels = np.array(labels).astype(np.float32)
		
		return images, labels

					# print (boxes)
					# img = cv2.resize(img,(448,448))
					# cv2.rectangle(img,(int(boxes[0]-boxes[2]/2),int(boxes[1]-boxes[3]/2)),(int(boxes[0]+boxes[2]/2),int(boxes[1]+boxes[3]/2)),(0,255,0),3)
					# cv2.imshow('test', img)
					# cv2.waitKey(300)


class udacity(object):
	def __init__(self):
		self.datapath = 'data/object-dataset/'
		self.labelpath = 'data/object-dataset/labels.csv'
		self.classes = ['"truck"', '"car"', '"pedestrian"', '"trafficLight"']
		# self.classinterest = [0, 6, 14, 2]
		self.classinterest = [0,1,2,3]
		# self.classinterest = list(range(20))
		# ['car', 'bicycle', 'bus', 'motorbike', 'person']
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE

	def load_udacity(self):
		import csv
		imname = None
		images = []
		labels = []
		with open(self.labelpath, newline='') as f:
			reader = csv.reader(f)
			r = 0
			ri = 0
			for row in reader:
				row = row[0].split(' ')
				r += 1
				if imname is None:
					imname = ''
					pass
				else:
					if imname == '':
						imname = self.datapath + row[0]
						img = cv2.imread(imname)
						h_img = img.shape[0]
						w_img = img.shape[1]
						h_ratio = 1.0 * self.image_size / img.shape[0]
						w_ratio = 1.0 * self.image_size / img.shape[1]
						label = np.zeros((self.cell_size, self.cell_size, 5+20))
					else:
						if self.datapath + row[0] == imname:
							pass
						else:
							images += [imname]
							labels += [label]
							ri += 1
							print (imname, label.shape, r,'/93086', ri, end='\r')
							imname = self.datapath + row[0]
							img = cv2.imread(imname)
							label = np.zeros((self.cell_size, self.cell_size, 5+20))
					# x1 = int(row[0])
					# x2 = int(row[2])
					# y1 = int(row[1])
					# y2 = int(row[3])

					x1 = (float(row[1]) - (w_img/2)) * w_ratio + (224)
					y1 = (float(row[2]) - (h_img/2)) * h_ratio + (224)
					x2 = (float(row[3]) - (w_img/2)) * w_ratio + (224)
					y2 = (float(row[4]) - (h_img/2)) * h_ratio + (224)
					class_name = row[6]
					for i in range(4):
						if class_name == self.classes[i]:
							cls_ind = self.classinterest[i]
					boxes = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]
					x_ind = int(boxes[0] * self.cell_size / self.image_size)
					y_ind = int(boxes[1] * self.cell_size / self.image_size)
					if label[y_ind, x_ind, 0] == 1:
						continue
					label[y_ind, x_ind, 0] = 1
					label[y_ind, x_ind, 1:5] = boxes
					label[y_ind, x_ind, 5 + cls_ind] = 1

		images = np.array(images)
		labels = np.array(labels).astype(np.float32)
		
		return images, labels


def main():
	# print ('load voc')
	# voc_d = voc()
	# voc_images, voc_labels = voc_d.load_pascal()
	# print ('')
	print ('load crowdai')
	crowdai_d = crowdai()
	cai_images, cai_labels = crowdai_d.load_crowdai()
	print ('')
	print ('load udacity')
	udacity_d = udacity()
	udc_images, udc_labels = udacity_d.load_udacity()

	# vc_images = np.concatenate((voc_images,cai_images), axis=0)
	# images = np.concatenate((vc_images,udc_images), axis=0)
	# print (images.shape)
	# vc_labels = np.concatenate((voc_labels,cai_labels), axis=0)
	# labels = np.concatenate((vc_labels,udc_labels), axis=0)
	# print (labels.shape)

	images = np.concatenate((udc_images,cai_images), axis=0)
	labels = np.concatenate((udc_labels,cai_labels), axis=0)

	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, random_state = 42)
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)

	pickle_packet = {'xtrain' : X_train,
					 'ytrain' : y_train,
					 'xvalid' : X_val,
					 'yvalid' : y_val,
					 'xtest'  : X_test,
					 'ytest'  : y_test}
	pickle_name = 'data/Pickle/' + 'train_4_' + str(images.shape[0]) + '.pkl'

	# if os.path.exists(pickle_name):
	with open(pickle_name, 'wb') as f:
		pickle.dump(pickle_packet, f)
		print (pickle_name + ' saved')

if __name__=='__main__':    
	# d = crowdai()
	# d.load_crowdai()
	# d.get()

	# d = udacity()
	# d.load_udacity()

	main()
