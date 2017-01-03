#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for thestrroductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

import itertools
import csv 
import matplotlib.pyplot as plt
import time

#matplotlib.use('TkAgg')
plt.switch_backend('agg')
#from dataset_tools import load_labeled_data_train, load_labeled_data_test
from load_images import load_training_data, load_test_data
#from googlenet import build_model

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading itstro numpy arrays. It doesn't involve Lasagne at all.

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(param_list,input_var):
	# As a third model, we'll create a CNN of two convolution + pooling stages
	# and a fully-connected hidden layer in front of the output layer.

	#Default input layer, to be overwritten
	network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
						
	for param in param_list:
		if param[0] == "input":
			network = lasagne.layers.InputLayer(shape=(None, param[1], param[2], param[2]),input_var=input_var)
		elif param[0] == "dropout":
			network = lasagne.layers.dropout(network, p=param[1])
		elif param[0] == "dense":
			network = lasagne.layers.DenseLayer(network, num_units = param[1],nonlinearity=lasagne.nonlinearities.rectify)
		elif param[0] == "output":
			network = lasagne.layers.DenseLayer(network,num_units=param[1],nonlinearity=lasagne.nonlinearities.softmax)
	
	return network

def build_cnn(param_list,input_var):
	# As a third model, we'll create a CNN of two convolution + pooling stages
	# and a fully-connected hidden layer in front of the output layer.

	#Default input layer, to be overwritten
	network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
						
	for param in param_list:
		if param[0] == "input":
			network = lasagne.layers.InputLayer(shape=(None, param[1], param[2], param[2]),input_var=input_var)
		elif param[0] == "conv":
			network = lasagne.layers.Conv2DLayer(network, num_filters=param[1], filter_size=(param[2], param[2]),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
		elif param[0] == "pool":
			network = lasagne.layers.MaxPool2DLayer(network, pool_size=(param[1], param[1]))
		elif param[0] == "dropout":
			network = lasagne.layers.dropout(network, p=param[1])
		elif param[0] == "dense":
			network = lasagne.layers.DenseLayer(network, num_units = param[1],nonlinearity=lasagne.nonlinearities.rectify)
		elif param[0] == "output":
			network = lasagne.layers.DenseLayer(network,num_units=param[1],nonlinearity=lasagne.nonlinearities.softmax)
	
	return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train(param_list, hyperparam_list, model, architecture_name=''):
	
	print("Training " + architecture_name)
	
	num_epochs = hyperparam_list[0]
	learning_rate = hyperparam_list[1]
	momentum = hyperparam_list[2]

	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val = load_training_data()
	X_test, y_test = load_test_data()

	y_train -= 1
	y_val -= 1

	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')
	
	test_results = [["Filter Size","Number of Filters","Max Pool Size","Dropout","Dense Layer Number of Units"]]
	# Create neural network model (depending on first command line parameter)
	print("Building model and compiling functions...")
	if model == 'cnn':
		network = build_cnn(param_list,input_var)
	#elif model == 'googlenet':
		#network = build_model(input_var)['prob']
	elif model == 'mlp':
		network = build_mlp(param_list,input_var)
	else:
		print("Unrecognized model type %r." % model)
		return

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate=learning_rate, momentum=momentum)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
						dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True, on_unused_input='warn')

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True, on_unused_input='warn')


	# This function added by JACOB FUCkin RITCHIE
	# to get the predicted output

	predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	
	val_acc_list = []	
	val_err_list = []
	train_acc_list = []
	train_err_list = []
	
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		train_acc = 0
                
		for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			#_, acc = val_fn(inputs,targets)
			#train_acc += acc
			
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("	 training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("	 validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("	 validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))
		val_acc_list.append(val_acc / val_batches * 100)
		val_err_list.append(val_err / val_batches)
		#train_acc_list.append(train_acc / train_batches * 100)
		train_err_list.append(train_err / train_batches)

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	
	prediction_train_list = []
	prediction_val_list = []
	prediction_test_list = []
	
	for batch in iterate_minibatches(X_test, y_test, 1, shuffle=False):
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
		prediction_test_list.extend(predict_fn(inputs).tolist())
	prediction_test_vector	= np.array(prediction_test_list)
	train_target_vector = []
	for batch in iterate_minibatches(X_train, y_train, 1, shuffle=False):
		inputs, targets = batch
		prediction_train_list.extend(predict_fn(inputs).tolist())
		train_target_vector.append(targets)
	prediction_train_vector	 = np.array(prediction_train_list)
	
	for batch in iterate_minibatches(X_val, y_val, 1, shuffle=False):
		inputs, targets = batch
		prediction_val_list.extend(predict_fn(inputs).tolist())
	prediction_val_vector	 = np.array(prediction_val_list)
	print("Final results:")
	print("	 test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("	 test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

	x = np.array(val_acc_list)
	y = np.array(val_err_list)
	z = np.array(train_err_list)
	w = np.array(range(1,num_epochs+1))

	stamp = time.strftime("%Y%m%d_%H%M%S")
	np.savez("./results/training_curve_dump"+ stamp + architecture_name + ".npz", epoch=w, val_acc=x, val_err=y,train_err=z)

	plt.plot(range(1,num_epochs+1),val_acc_list)
	#plt.plot(range(1,num_epochs+1),test_acc_list, 'r')
	plt.xlabel('Epochs')
	plt.ylabel('Validation Accuracy')
	plt.legend()
	plt.title(architecture_name)
	plt.ylim([0,100])
	plt.savefig('./val_acc'+ stamp	 + architecture_name + '.png')
	plt.clf()	

	plt.plot(range(1,num_epochs+1),val_err_list, label = 'Validation Error')
	plt.plot(range(1,num_epochs+1),train_err_list, 'r', label = 'Training Error')
	plt.xlabel('Epochs')
	plt.ylabel('Cross Entropy')
	plt.legend()
	plt.title(architecture_name)
	plt.ylim([0,3])
	plt.savefig('./results/val_err'+ stamp + architecture_name + '.png')
	plt.ylim([0,2])
	plt.clf()
	
	with open('./results/test_prediction_'+ stamp + architecture_name + '.csv','w') as f:
		f.write("Id,Prediction\n")
		for i in range(2970):
			if i <= 969:
					f.write(str(i+1) + "," + str(prediction_test_vector[i] + 1)  + "\n")
			else:
					f.write(str(i+1) + ",0\n")
		f.close()
	
	with open('./results/val_prediction_'+ stamp + architecture_name + '.csv','w') as f:
		f.write("Id,Prediction\n")
		for i in range(len(prediction_val_vector)):
			f.write(str(i+1) + "," + str(prediction_val_vector[i] + 1)  + "\n")
		f.close()	 

	with open('./results/train_prediction_'+ stamp + architecture_name + '.csv','w') as f:
		f.write("Id,Prediction\n")
		for i in range(len(prediction_train_vector)):
			f.write(str(i+1) + "," + str(prediction_train_vector[i] + 1)  + "\n")
		f.close()			 
	with open('/home/zhaowen9/hw3/results/train_target_vector'+ stamp + architecture_name + '.csv','w') as f:
		f.write("Id,Prediction\n")
		for i in range(len(train_target_vector)):
			f.write(str(i+1) + "," + str(train_target_vector[i]) + "\n")
		f.close()	
	return test_acc / test_batches * 100

		# Optionally, you could now dump the network weights to a file like this:
		# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
		#
		# And load them again later on like this:
		# with np.load('model.npz') as f:
		#		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		# lasagne.layers.set_all_param_values(network, param_values)
	

if __name__ == '__main__':
	
	#for x in itertools.product(filter_size_list,num_filters_list,pool_size_list,dropout_list,dense_num_units_list):
	test_results = [["paramlist","[#epocs, learningrate, momentum]", "results"]]
	hyperparam_list = [2,0.0001,0.9]
	

	#param_list = [("input",3,128),("conv",128,3),("pool",2),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("dropout",.5),("output",8)]
	#hyperparam_list = [200,0.0001,0.9]
	#results = round(train(param_list, hyperparam_list,"cnn"),2)

	#param_list = [("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("dropout",.5),("output",8)]
	
	#sweep_list = [["cnn_4_layer_2_dropout",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("dropout",.5),("output",8)]]]
					
					#["_1_layer",[("input",3,128),("conv",128,3),("pool",2),("dense",256),("output",8)]],
					#["_2_layer",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("dense",256),("output",8)]],
					#["_3_layer",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dense",256),("output",8)]],
					#["_4_layer",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dense",256),("output",8)]],
					#["_3_layer_1_dropout",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("output",8)]],
					#["_4_layer_1_dropout",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("conv",64,2),("pool",2),("dense",256),("output",8)]],
					#["_3_layer_2_dropout",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("dropout",.5),("output",8)]],
					#["_4_layer_2_dropout",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("dropout",.5),("output",8)]],
					#["_4_layer_3_dropout",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("dropout",.5),("output",8)]]
	
	sweep_list = [["cnn_4_layer_2_dropout",[("input",3,128),("conv",128,3),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("conv",64,2),("pool",2),("dropout",.5),("dense",256),("dropout",.5),("output",8)]]]		
	for architecture_name,param_list in sweep_list:
			results = round(train(param_list, hyperparam_list,"cnn",architecture_name=architecture_name),2)



	#with open("/home/zhaowen9/hw3/results/results"+time.strftime("%Y%m%d_%H%M%S")+".csv", "wb") as f:
	#	writer = csv.writer(f)
	#	writer.writerows(test_results)
	#print("Saving to file")
	
	
