import numpy as np
import scipy
import pdb
import sklearn
from sklearn import linear_model
from data_utils import *
import random
import matplotlib.pyplot as plt
import unirep
from unirep import *
from data_utils import *
import jax_unirep
from jax_unirep import get_reps
from jax_unirep.utils import load_params_1900

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

		X.append(get_reps(S[i], evotuned_weights)[0])
		print(i)

	return np.array(X)[:,0,:] # return the eUniRep sequence representation vector/vectors


# function for introducing mutations to a given sequence
def mutate_sequence(seq,m): # produce a mutant sequence (integer representation), given an initial sequence and the number of mutations to introduce ("m")

	for i in range(m): #iterate through number of mutations to add

		rand_loc = random.randint(132,212) # find random position to mutate
		rand_aa = random.randint(1,23) # find random amino acid to mutate to
		seq = list(seq)
		seq[rand_loc] = get_int_to_aa()[rand_aa] # update sequence to have new amino acid at randomely chosen position
		seq = ''.join(seq)


	return seq # output the randomely mutated sequence


# Directed-evolution function: 
def directed_evolution(s_wt,num_iterations,T,Model): # input = (wild-type sequence, number of mutation iterations, "temperature")		


	s_traj = [] # initialize an array to keep records of the protein sequences for this trajectory
	y_traj = [] # initialize an array to keep records of the fitness scores for this trajectory


	s = mutate_sequence(s_wt, (np.random.poisson(2) + 1)) # initial mutant sequence for this trajectory, with m = Poisson(2)+1 mutations
	x = get_eUniRep(np.array([s])) # eUniRep representation of the initial mutant sequence for this trajectory
	y = Model.predict(x) # predicted fitness score for the initial mutant sequence for this trajectory

	for i in range(num_iterations): # iterate through the trial mutation steps for the directed evolution trajectory

		mu = np.random.uniform(1,2.5) # "mu" parameter for poisson function: used to control how many mutations to introduce
		m = np.random.poisson(mu-1) + 1 # how many random mutations to apply to current sequence

		s_new = mutate_sequence(s, m) # new trial sequence, produced from "m" random mutations
		x_new = get_eUniRep(np.array([s_new])) # new eUniRep representation for trial sequence
		y_new = Model.predict(x_new) # new fitness value for trial sequence

		p = min(1,np.exp((y_new-y)/T)) # probability function for trial sequence
		if random.random() < p: # metropolis-Hastings update selection criteria
			s, y = s_new, y_new # if criteria is met, update sequence and corresponding fitness

		s_traj.append(s) # update the sequence trajectory records for this iteration of mutagenesis
		y_traj.append(y) # update the fitness trajectory records for this iteration of mutagenesis


	return s_traj, y_traj # output = (sequence record for trajectory, fitness score recorf for trajectory)


def run_DE_trajectories(s_wt, X,Y, num_iterations, num_trajectories, Model):

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
	Model.fit(X_train,Y_train)
	T = 0.01 # "temperature": a mutagenesis parameter that controls rate of trial-mutation acceptance

	s_records = []
	y_records = []

	for i in range(num_trajectories): #iterate through however many mutation trajectories we want to sample
		s_traj, y_traj = directed_evolution(s_wt,num_iterations,T,Model) # call the directed evolution function, outputting the trajectory sequence and fitness score records

		s_records.append(s_traj) # update the sequence trajectory records for this full mutagenesis trajectory
		y_records.append(y_traj) # update the fitness trajectory records for this full mutagenesis trajectory

	s_records = np.array(s_records)
	y_records = np.array(y_records)

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




# loading data, processing data to eUniRep representation, and fitting RidgeCV model for predictive function
evotune_weight_directory_location = "/Users/andrew/ML_PETase_project/blac_unirep_global_init_1"
evotuned_weights = load_params_1900(evotune_weight_directory_location)

s_wt = get_fasta_letters('inputs/beta_lac.fasta') # load wild-type sequence (integer representation)

X = np.load('inputs/beta_lac_eUniRep_big.npy')

Y = np.load('inputs/beta_lac_fitness_big.npy')


X = X[:96,:]
Y = Y[:96]

Y = Y- min(Y)
Y = Y/max(Y)
Model = linear_model.RidgeCV(alphas = np.logspace(-4,4,9),normalize=True,cv=10)



# --------------------------------------------------------------------------------------
# -------------------------- commands for test functions -------------------------------
# --------------------------------------------------------------------------------------


# 1). To plot one quick validation test of model performance for the X and Y data loaded:
#			>>>   simple_validation_plot(X,Y,Model)

# 2). To run the validation test for fitting the model:
#			 >>>   plot_multiple_validation_tests(X,Y,Model)
#	

#  3).To run the actual DE script:
#			 >>>   s_records, y_records = run_DE_trajectories(s_wt,X,Y, 10, 2, Model)
# 	











