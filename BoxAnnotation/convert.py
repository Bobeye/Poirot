import pickle
import cv2
import numpy as np 
from sklearn.model_selection import train_test_split
import glob
import os

path_to_images = 'Images/'
path_to_labels = 'Labels/'
path_to_data = 'Data/'
path_to_pickle = 'Pickle/'

NUM_CLASS = 20
IMG_X = 640
IMG_Y = 360
IMG_SIZE = 448
CELL_SIZE_X = 7
CELL_SIZE_Y = 7


X_SCALE = IMG_SIZE / IMG_X
Y_SCALE = IMG_SIZE / IMG_Y

def convert_to_labels(filename):
	objs = []
	label = np.zeros((CELL_SIZE_Y, CELL_SIZE_X, 5 + NUM_CLASS)).astype(np.float32)
	for i in range(NUM_CLASS):
		n_class = i
		path = path_to_labels + '%03d' % (i) + '/' + filename
		if os.path.exists(path):
			txtfile = open(path, 'r')
			objs = txtfile.readlines()
			objs = [obj.strip() for obj in objs]
			if int(objs[0]) == 0:
				pass
			else:
				for j in range(len(objs)-1):
					j += 1
					objs[j] = objs[j].split(' ')
					x1 = max((int(objs[j][0]) - (IMG_X/2)) * X_SCALE + (IMG_SIZE/2), 0) 
					y1 = max((int(objs[j][1]) - (IMG_Y/2)) * Y_SCALE + (IMG_SIZE/2), 0) 
					x2 = min((int(objs[j][2]) - (IMG_X/2)) * X_SCALE + (IMG_SIZE/2), IMG_SIZE)
					y2 = min((int(objs[j][3]) - (IMG_Y/2)) * Y_SCALE + (IMG_SIZE/2), IMG_SIZE) 
					boxes = [(x2 + x1) / 2, (y2 + y1) / 2, x2 - x1, y2 - y1]
					x_ind = int((x2 + x1) / 2 * CELL_SIZE_X / IMG_SIZE)
					y_ind = int((y2 + y1) / 2 * CELL_SIZE_Y / IMG_SIZE)
					if label[y_ind, x_ind, 0] == 1:
						continue
					label[y_ind, x_ind, 0] = 1
					label[y_ind, x_ind, 1:5] = boxes
					label[y_ind, x_ind, 5 + n_class] = 1
		else:
			pass
	return label
	

img_fils = glob.glob(path_to_images + '*.jpg')
images = []
labels = []
for f in img_fils:
	img = cv2.imread(f)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = (img / 255.0) * 2.0 - 1.0
	images += [img]
	lf = f.split('/')[1].split('.')[0] + '.txt'
	labels += [convert_to_labels(lf)]
images = np.array(images).astype(np.float32)
labels = np.array(labels).astype(np.float32)

print (images.shape)
print (labels.shape)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)

pickle_packet = {'xtrain' : X_train,
				 'ytrain' : y_train,
				 'xvalid' : X_val,
				 'yvalid' : y_val,
				 'xtest'  : X_test,
				 'ytest'  : y_test}
pickle_name = path_to_pickle + 'train' + str(images.shape[0]) + '.pkl'

# if os.path.exists(pickle_name):
with open(pickle_name, 'wb') as f:
	pickle.dump(pickle_packet, f)
	print (pickle_name + ' saved')

