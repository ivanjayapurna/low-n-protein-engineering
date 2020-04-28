import numpy as np
from data_utils import *
import random
import matplotlib.pyplot as plt
import jax_unirep
from jax_unirep import get_reps
from jax_unirep.utils import load_params_1900
import pdb
import scipy
from scipy.stats import uniform as sp_rand
from scipy.optimize import minimize
import sklearn
from sklearn import linear_model
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression,LogisticRegression,LassoLarsCV
from sklearn.model_selection import RandomizedSearchCV,train_test_split,cross_val_score, ShuffleSplit, cross_val_predict, KFold

from sklearn.metrics import make_scorer


from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor



def get_fasta_letters(filename): # given the fasta filename for the wild-type, turn it into a letter-representation of the amino acid sequence
    with open(filename, 'r') as f:
        seq = ""
        for line in f:
            if line[0] == '>' and not seq == "":
                seq = ""
            elif not line[0] == '>':
                seq += line.replace("\n","")
    return seq

def get_SY(seq_file,fitness_file): # load the training data, given the filepath to the .txt files that contain training sequences and their corresponding fitness scores
	S = np.loadtxt('inputs/'+seq_file, dtype='str') # S is an array representation of the amino acid sequences, currently stored as the letter-encoding
	Y = np.loadtxt('inputs/'+fitness_file,dtype='float') # Y is an array of the fitness scores for training data
	return S, Y # output the training data integer-sequences and their corresponding fitness scores

def shuffle(X,Y):
  indices = np.random.permutation(np.shape(X)[0])
  X,Y = X[indices], Y[indices]
  return X,Y

def get_eUniRep(S): # this is just a placeholder function until we get the function to produce the actual eUniRep representation of training sequences
	
	X = []
	for i in range(S.shape[0]):
		# pdb.set_trace()
		X.append(get_reps(S[i], evotuned_weights)[0])
		print(i)

	return np.array(X)[:,0,:] # return the eUniRep sequence representation vector/vectors




def validation_test(X,Y,Model):
	X,Y = shuffle(X,Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
	Model.fit(X_train,Y_train)

	preds = Model.predict(X_test)
	loss = calc_loss(Y_test,preds)

	return loss, Y_test, preds


def plot_multiple_validation_tests(X,Y,Model):

	loss_list = []

	Y_valid_data_list = []
	Y_valid_preds_list = []
	for i in range(20):

		loss, Y_test, preds = validation_test(X,Y,Model)
		loss_list.append(loss)
		Y_valid_data_list.extend(Y_test.tolist())
		Y_valid_preds_list.extend(preds.tolist())

	plt.figure()
	error_list = np.sqrt((np.array(Y_valid_data_list)-np.array(Y_valid_preds_list))**2)
	plt.errorbar(np.arange(len(Y_valid_data_list)),Y_valid_data_list,yerr=error_list, fmt='-o')

	plt.show()

def simple_validation_plot(X,Y,Model):
	# 

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

	Model.fit(X_train,Y_train)
	loss = np.linalg.norm(Model.predict(X_test)-Y_test)
	print(loss)

	Y_preds = Model.predict(X_test)
	Y_data = Y_test

	plt.figure()
	plt.scatter(np.arange(len(Y_data)),Y_preds)
	plt.scatter(np.arange(len(Y_data)),Y_data)
	plt.plot(np.arange(len(Y_data)),Y_preds)
	plt.plot(np.arange(len(Y_data)),Y_data)
	plt.title('loss = '+str(loss))
	plt.legend(['preds','data'])

	plt.show()

def distance_matrix(N):
	distance_matrix = np.zeros((N,N))
	for i in range(N):
		for j in range(N):
			# distance_matrix[i,j]=1- ((abs(i-j)/N)**2)
			distance_matrix[i,j]= 1-(abs(i-j)/N)

	return distance_matrix

def confusion_matrix_loss(Y_test,Y_preds_test):

	N = len(Y_test)
	Y_rank_matrix = np.zeros((N,N))
	Y_preds_rank_matrix = np.zeros((N,N))

	for i in range(N):
		for j in range(N):

			if Y_test[i] > Y_test[j]:
				Y_rank_matrix[i,j] = 1
			elif Y_test[i] <= Y_test[j]:
				Y_rank_matrix[i,j] = 0


			if Y_preds_test[i] > Y_preds_test[j]:
				Y_preds_rank_matrix[i,j] = 1
			elif Y_preds_test[i] <= Y_preds_test[j]:
				Y_preds_rank_matrix[i,j] = 0

	confusion_matrix = ~(Y_preds_rank_matrix == Y_rank_matrix)

	dist_mat = distance_matrix(N)
	confusion_matrix = confusion_matrix*dist_mat


	loss = np.sum(confusion_matrix)/confusion_matrix.size
	# loss = np.sum(confusion_matrix)

	# pdb.set_trace()
	return loss

def objective_function(alpha, X, Y):
	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
	# rcv = RidgeCV(alphas = alphas,normalize=True,gcv_mode='eigen', cv=10).fit(X, Y)
	rcv = Ridge(alpha = alpha,normalize=True,solver="auto").fit(X_train, Y_train)
	error_rank = confusion_matrix_loss(Y_test,rcv.predict(X_test))
	error_mse = np.sum((Y_test-rcv.predict(X_test))**2)
	# print(error_rank,error_mse)
	print(alpha,error_rank)
	# error = loss_function(np.matmul(X,beta), Y)
	return error_rank



S,Y = get_SY("new_mut_seqs.txt","new_fitness_data.txt")

s_wt = np.array([get_fasta_letters('inputs/GAP38373.1.fasta')]) # load wild-type sequence (integer representation)
y_wt = np.array([0.0])
evotune_weight_directory = '/Users/andrew/ML_PETase_project/PETase_weights_1E5_epoch4'
evotuned_weights = load_params_1900(evotune_weight_directory)


x_wt = np.load("inputs/"+"PETase_xwt_eUniRep_LR1E5_epoch4.npy")
X = np.load("inputs/"+"PETase_Xmuts_eUniRep_LR1E5_epoch4.npy")
X = np.concatenate((x_wt,X))
Y = np.concatenate((y_wt,Y))
# pdb.set_trace()

# X = np.load('inputs/beta_lac_eUniRep_big.npy')
# # X = sklearn.preprocessing.normalize(X)
# Y = np.load('inputs/beta_lac_fitness_big.npy')

X_mean = np.mean(X,axis=0)
X= (X-X_mean)
X_norm = np.linalg.norm(X,axis=0)
X= X/X_norm
# X= (X-X_mean)/X_norm

Y_mean = np.mean(Y)
Y_norm = np.linalg.norm(Y)
Y = (Y-Y_mean)/Y_norm
# Y_std = np.std(Y)
# Y = (Y-Y_mean)/Y_std
# Y = Y- min(Y)
# Y = Y/max(Y)

X_mixed,Y_mixed = shuffle(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

alpha = 0.5

kfold = KFold(n_splits=10, random_state=42, shuffle=True)


model = RidgeCV(cv=kfold)


model.fit(X_train, Y_train)

Y_train_preds = model.predict(X_train)
Y_test_preds = model.predict(X_test)

train_loss = confusion_matrix_loss(Y_train,Y_train_preds)
test_loss = confusion_matrix_loss(Y_test,Y_test_preds)
print(alpha,train_loss,test_loss)

plt.figure()
plt.plot(Y_train[np.argsort(Y_train)])
plt.plot(Y_train_preds[np.argsort(Y_train)])

plt.legend(['Y_train','Y_train preds'])

plt.figure()
plt.plot(Y_test[np.argsort(Y_test)])
plt.plot(Y_test_preds[np.argsort(Y_test)])

plt.legend(['Y_test','Y_test pred'])

plt.show()


