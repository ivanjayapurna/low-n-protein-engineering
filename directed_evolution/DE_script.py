import numpy as np
import scipy
import pdb
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoLarsCV
from data_utils import *
import pandas as pd
import random
import matplotlib.pyplot as plt
from unirep import babbler1900 as babbler
import unirep
from unirep import *
import data_utils
import tensorflow as tf
import jax_unirep
from jax_unirep import get_reps
import time

from sklearn.model_selection import train_test_split

def get_fasta_letters(filename): # given the fasta filename for the wild-type, turn it into a letter-representation of the amino acid sequence
    with open(filename, 'r') as f:
        seq = ""
        for line in f:
            if line[0] == '>' and not seq == "":
                seq = ""
            elif not line[0] == '>':
                seq += line.replace("\n","")
    return seq

def get_SY(): # load the training data, given the filepath to the .txt files that contain training sequences and their corresponding fitness scores
	S = np.loadtxt('inputs/wt_mutants.txt', dtype='str') # S is an array representation of the amino acid sequences, currently stored as the letter-encoding
	Y = np.loadtxt('inputs/fitness.txt',dtype='float') # Y is an array of the fitness scores for training data
	return S, Y # output the training data integer-sequences and their corresponding fitness scores

def shuffle(X,Y):
  indices = np.random.permutation(np.shape(X)[0])
  X,Y = X[indices], Y[indices]
  return X,Y

def get_eUniRep(S): # this is just a placeholder function until we get the function to produce the actual eUniRep representation of training sequences
	
	X = []
	for i in range(S.shape[0]):

		X.append(get_reps(S[i])[0])

	return np.array(X)[:,0,:] # return the eUniRep sequence representation vector/vectors


# function for introducing mutations to a given sequence
def mutate_sequence(seq,m): # produce a mutant sequence (integer representation), given an initial sequence and the number of mutations to introduce ("m")

	for i in range(m): #iterate through number of mutations to add
		rand_loc = random.randint(0,(len(seq)-1)) # find random position to mutate
		rand_aa = random.randint(1,25) # find random amino acid to mutate to
		seq[rand_loc] = rand_aa # update sequence to have new amino acid at randomely chosen position

	return seq # output the randomely mutated sequence


# Directed-evolution function: 
def directed_evolution(s_wt,num_iterations,T): # input = (wild-type sequence, number of mutation iterations, "temperature")		

	s_traj = np.zeros((num_iterations,len(s_wt))) # initialize an array to keep records of the protein sequences for this trajectory
	y_traj = np.zeros((num_iterations,1)) # initialize an array to keep records of the fitness scores for this trajectory

	s = mutate_sequence(s_wt, (np.random.poisson(2) + 1)) # initial mutant sequence for this trajectory, with m = Poisson(2)+1 mutations
	# x = eUniRep_placeholder(np.array(s)) # eUniRep representation of the initial mutant sequence for this trajectory
	x = get_eUniRep(s) # eUniRep representation of the initial mutant sequence for this trajectory
	y = model_handle(x) # predicted fitness score for the initial mutant sequence for this trajectory

	for i in range(num_iterations): # iterate through the trial mutation steps for the directed evolution trajectory

		mu = np.random.uniform(1,2.5) # "mu" parameter for poisson function: used to control how many mutations to introduce
		m = np.random.poisson(mu-1) + 1 # how many random mutations to apply to current sequence

		s_new = mutate_sequence(s, m) # new trial sequence, produced from "m" random mutations
		# x_new = eUniRep_placeholder(np.array(s_new)) # new eUniRep representation for trial sequence
		x_new = get_eUniRep(s_new) # new eUniRep representation for trial sequence
		y_new = model_handle(x_new) # new fitness value for trial sequence

		p = min(1,np.exp((y_new-y)/T)) # probability function for trial sequence
		if random.random() < p: # metropolis-Hastings update selection criteria
			s, y = s_new, y_new # if criteria is met, update sequence and corresponding fitness

		s_traj[i] = s # update the sequence trajectory records for this iteration of mutagenesis
		y_traj[i] = y # update the fitness trajectory records for this iteration of mutagenesis

	return s_traj, y_traj # output = (sequence record for trajectory, fitness score recorf for trajectory)


def run_DE_trajectories(s_wt, num_iterations, num_trajectories, model_handle):

	num_trajectories = 10 # define number of trajectories to sample
	num_iterations = 3000 # define number of trial mutagenesis steps per trajecory
	T = 0.01 # "temperature": a mutagenesis parameter that controls rate of trial-mutation acceptance

	s_records = np.zeros((num_trajectories, num_iterations, len(s_wt)))
	y_records = np.zeros((num_trajectories, num_iterations, 1))

	for i in range(num_trajectories): #iterate through however many mutation trajectories we want to sample
		s_traj, y_traj = directed_evolution(s_wt,num_iterations,T) # call the directed evolution function, outputting the trajectory sequence and fitness score records

		s_records[i] = s_traj # update the sequence trajectory records for this full mutagenesis trajectory
		y_records[i] = y_traj # update the fitness trajectory records for this full mutagenesis trajectory


	plt.plot(np.transpose(y_records[:,:,0])) # plot the changes in fitness for all sampled trajectories
	plt.ylabel(r'$\Delta T_m$')
	plt.xlabel('Mutation Steps')
	plt.show() # cuz it is very satisfying to visualize this stuff :)

	return s_records, y_records


def shuffle(X,Y):
	indices = np.random.permutation(np.shape(X)[0])
	X,Y = X[indices], Y[indices]
	return X,Y


def calc_loss(Y_test,preds):
	return np.linalg.norm(Y_test-preds)

def validation_test(X,Y):
	X,Y = shuffle(X,Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
	Model = RidgeCV(alphas=1e3,normalize=True,cv=10).fit(X_train, Y_train) 

	preds = Model.predict(X_test)
	loss = calc_loss(Y_test,preds)
	print('alpha: ', Model.alpha_, '  loss: ', loss)

	return loss, Y_test[:,0], preds, Model.alpha_


def plot_multiple_validation_tests(X,Y):

	loss_list = []
	alpha_list = []
	Y_valid_data_list = []
	Y_valid_preds_list = []
	for i in range(20):
		loss, Y_test, preds ,alpha= validation_test(X,Y)
		loss_list.append(loss)
		alpha_list.append(alpha)
		Y_valid_data_list.extend(Y_test.tolist())
		Y_valid_preds_list.extend(preds.tolist())
	Y_valid_data_list, Y_valid_preds_list = (list(t) for t in zip(*sorted(zip(Y_valid_data_list, Y_valid_preds_list))))
	plt.figure()
	plt.plot(Y_valid_data_list,c='cyan')
	plt.plot(Y_valid_preds_list,c='magenta')
	plt.legend(['Data','Preds'])
	plt.ylabel('Fitness Score')
	plt.xlabel('Sorted mutants from random validation tests')



	plt.figure()
	plt.scatter(alpha_list,loss_list)
	plt.xlabel('alpha')
	plt.xscale("log")
	plt.ylabel('loss')
	plt.show()



# loading data, processing data to eUniRep representation, and fitting RidgeCV model for predictive function
s_wt = get_fasta_letters("inputs/GAP38373.1.fasta") # load wild-type sequence (integer representation)
y_wt = 0.0 # call corresponding wild-type fitness score
S, Y = get_SY() # load training sequences (S, integer representation) and their corresponding fitness scores (Y)
x_wt = get_eUniRep(np.array([s_wt]))
X = get_eUniRep(S)
Y = Y[:,np.newaxis]
Model = RidgeCV(normalize=True, cv=10).fit(X, Y) 
model_handle = lambda x: Model.predict(x[np.newaxis,:]) # create a function-handle for the regression model for convenient fitness score prediction in mutation steps


# --------------------------------------------------------------------------------------
# -------------------------- commands for test functions -------------------------------
# --------------------------------------------------------------------------------------
# 1).  To run the actual DE script:
#		 >>>   s_records, y_records = run_DE_trajectories(s_wt, num_iterations, num_trajectories, model_handle)
# 
# 2). To run the validation test for fitting the model:
#		 >>>   plot_multiple_validation_tests(X,Y)
# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------













