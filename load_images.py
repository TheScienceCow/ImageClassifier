import matplotlib.image as img
import numpy as np
import os,csv,sys
from random import shuffle

def get_labels(filename):
	labels = {}
	with open(filename, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			next(reader,None)
			for row in reader:
				labels[int(row[0])] = int(row[1])
	return labels


def load_training_data():
	all_train_data,train_data,validation_data = [],[],[]
	all_train_label,train_label,validation_label = [],[],[]
	labels_filename = 'train.csv'
	labels = get_labels(labels_filename)
	for x in os.walk('./train'):
		#print x
		#sys.exit()

		print "ITERATION"
		for filename in	 x[2]:

				image = img.imread("./train/" + filename)
				image_id = int(filename.lstrip("0").split(".")[0])
				all_train_data.append(image)
				label = labels[image_id]
				all_train_label.append(label)

	all_train_data_shuffled = []
	all_train_label_shuffled = []
	index_shuf = range(len(all_train_label))
	for i in index_shuf:
		all_train_data_shuffled.append(all_train_data[i])
		all_train_label_shuffled.append(all_train_label[i])
	num_train = int(len(all_train_label_shuffled) * 0.25)
	low_frequency_classes = [3,4,6,7,8]
	for i in range(num_train,len(all_train_label_shuffled)):
			label = all_train_label_shuffled[i]
			if label in low_frequency_classes:
				all_train_label_shuffled.append(label)
				all_train_data_shuffled.append(all_train_data_shuffled[i])
				



	
	train_data.extend(all_train_data_shuffled[num_train:])
	validation_data.extend(all_train_data_shuffled[:num_train])
	train_label.extend(all_train_label_shuffled[num_train:])
	validation_label.extend(all_train_label_shuffled[:num_train])


	X_train = np.transpose(np.array(train_data), (0, 3, 1, 2))
	y_train = np.array(train_label)
	X_val = np.transpose(np.array(validation_data), (0, 3, 1, 2))
	y_val = np.array(validation_label)
	print(X_train.shape)
	print(y_train.shape)
	print(X_val.shape)
	print(y_val.shape)
	return X_train, y_train, X_val, y_val

def load_test_data():
	test_data,test_label = [],[]

	for x in os.walk('./val'):
		for filename in	 x[2]:
				image = img.imread("./val/" + filename)
				test_data.append(image)
				label = 0
				test_label.append(label)

	X_test = np.transpose(np.array(test_data), (0, 3, 1, 2))
	y_test = np.array(test_label)

	return X_test, y_test
	pass


def load_training_data_old():
	image_0 = []
	image_1 = []
	image_2 = []
	image_3 = []

	label_0 = []
	label_1 = []
	label_2 = []
	label_3 = []
	
	train_data = []
	validation_data = []
	train_label = []
	validation_label = []
	
	print('Loading training data set from /data/anne/cellularity/Digit_train/')
	for x in os.walk('/data/anne/cellularity/Digit_train/'):
		#print(x)
		if x[0].endswith('0'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_train/0/'+files)
				image_0.append(image)
				label_0.append(0)
		if x[0].endswith('1'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_train/1/'+files)
				image_1.append(image)
				label_1.append(1)
		if x[0].endswith('2'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_train/2/'+files)
				image_2.append(image)
				label_2.append(2)
		if x[0].endswith('3'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_train/3/'+files)
				image_3.append(image)
				label_3.append(3)
				
				
	num_train_0 = int(len(image_0) * 0.75)
	num_train_1 = int(len(image_1) * 0.75)
	num_train_2 = int(len(image_2) * 0.75)
	num_train_3 = int(len(image_3) * 0.75)

	train_data.extend(image_0[:num_train_0])
	validation_data.extend(image_0[num_train_0:])
	train_label.extend(label_0[:num_train_0])
	validation_label.extend(label_0[num_train_0:])

	train_data.extend(image_1[:num_train_1])
	validation_data.extend(image_1[num_train_1:])
	train_label.extend(label_1[:num_train_1])
	validation_label.extend(label_1[num_train_1:])

	train_data.extend(image_2[:num_train_2])
	validation_data.extend(image_2[num_train_2:])
	train_label.extend(label_2[:num_train_2])
	validation_label.extend(label_2[num_train_2:])

	train_data.extend(image_3[:num_train_3])
	validation_data.extend(image_3[num_train_3:])
	train_label.extend(label_3[:num_train_3])
	validation_label.extend(label_3[num_train_3:])

			
	X_train = np.transpose(np.array(train_data), (0, 3, 1, 2))
	y_train = np.array(train_label)
	X_val = np.transpose(np.array(validation_data), (0, 3, 1, 2))
	y_val = np.array(validation_label)

	return X_train, y_train, X_val, y_val
	
def load_test_data_old():
	image_0 = []
	image_1 = []
	image_2 = []
	image_3 = []

	label_0 = []
	label_1 = []
	label_2 = []
	label_3 = []
	
	test_data = []
	test_label = []
	
	print('Loading test data set from /data/anne/cellularity/Digit_test/')
	for x in os.walk('/data/anne/cellularity/Digit_test/'):
		#print(x)
		if x[0].endswith('0'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_test/0/'+files)
				image_0.append(image)
				label_0.append(0)
		if x[0].endswith('1'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_test/1/'+files)
				image_1.append(image)
				label_1.append(1)
		if x[0].endswith('2'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_test/2/'+files)
				image_2.append(image)
				label_2.append(2)
		if x[0].endswith('3'):
			for files in x[2]:
				image = img.imread('/data/anne/cellularity/Digit_test/3/'+files)
				image_3.append(image)
				label_3.append(3)
				
				
	test_data.extend(image_0)
	test_label.extend(label_0)

	test_data.extend(image_1)
	test_label.extend(label_1)

	test_data.extend(image_2)
	test_label.extend(label_2)

	test_data.extend(image_3)
	test_label.extend(label_3)

			
	X_test = np.transpose(np.array(test_data), (0, 3, 1, 2))
	y_test = np.array(test_label)

	return X_test, y_test

def main():
	X_test, y_test = load_test_data()
	X_train, y_train, X_val, y_val = load_training_data()
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)

if __name__ == '__main__':
	main()
