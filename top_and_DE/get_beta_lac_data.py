import pandas as pd 
import numpy as np 
import pdb
from data_utils import *

def get_mut_letter_seq(wt_seq, fitness_scores,  mutation_keys):
	mut_seq_letter_list = []
	# mut_seq_int_list = []
	mut_fitness_list = []
	# pdb.set_trace()
	for mut_id in mutation_keys:
		# print(mut_id)
		mut_seq = list(wt_seq)
		position_index = int(mut_id[1:-1])-1
		wt_aa = mut_id[0]
		mut_aa = mut_id[-1]
		
		if wt_seq[position_index] == wt_aa:
			mut_seq[position_index] = mut_aa
		else:
			print("UH ohh we gotta problem! weeeooohhh! weeooohhh!")

		mut_letter_seq = ''.join(mut_seq)
		mut_seq_letter_list.append(mut_letter_seq)
		# mut_seq_int_list.append(aa_seq_to_int(mut_letter_seq))
		# pdb.set_trace()
		mut_fitness_list.append(fitness_scores[np.where(mutation_keys==mut_id)[0]][0])
		# 
		
		# mut_seq = 
	# return mut_seq_letter_list,mut_seq_int_list,mut_dTm_list
	return mut_seq_letter_list,mut_fitness_list


def write_seq_text(mut_seq_letter_list,output_filename):

	output_filename = os.path.join(output_filename)
	with open(output_filename, 'w') as file:
		seq = ""
		for line in mut_seq_letter_list:
			file.write(line + '\n')



def get_fasta_letters(filename):
    # I don't know exactly how to do this in tf, so resorting to python.
    # Should go line by line so everything is not loaded into memory
    
    # sourcefile = os.path.join(source)
    # destination = os.path.join(destination)
    with open(filename, 'r') as f:
        # with open(destination, 'w') as dest:
        seq = ""
        for line in f:
            if line[0] == '>' and not seq == "":
                # dest.write(aas_to_int_seq(seq) + '\n')
                seq = ""
            elif not line[0] == '>':
                seq += line.replace("\n","")
    return seq


data_sheet = pd.read_excel('beta_lac_fitness_data.xlsx').to_numpy()

# ambler_locs = data_sheet[:,0]



# non_start = ambler_locs>3
# non_stop = ambler_locs<291
# coding_indices = non_start*non_stop
# coding_indices = ambler_locs<291

# aa_locs = ambler_locs[coding_indices]-3
# wt_AA = data_sheet[coding_indices,1]
# mut_AA = data_sheet[coding_indices,2]
# counts = data_sheet[coding_indices,3]
# fitness = data_sheet[coding_indices,4]
# error = data_sheet[coding_indices,5]


ambler_locs = data_sheet[:,0]
wt_AA = data_sheet[:,1]
mut_AA = data_sheet[:,2]
counts = data_sheet[:,3]
fitness = data_sheet[:,4]
error = data_sheet[:,5]


coding_locations = wt_AA != "*"
wt_AA = wt_AA[coding_locations]
mut_AA =mut_AA[coding_locations]
counts =counts[coding_locations]
fitness =fitness[coding_locations]
error = error[coding_locations]
AA_positions = (np.arange(wt_AA.shape[0])//20)+1

valid_fitness_indices = ((error/fitness)<0.15)

wt_AA= wt_AA[valid_fitness_indices]
mut_AA= mut_AA[valid_fitness_indices]
fitness= fitness[valid_fitness_indices]
AA_positions = AA_positions[valid_fitness_indices]

indices = np.random.permutation(np.shape(wt_AA)[0])

# pdb.set_trace()

indices = indices[:int(indices.shape[0]*0.06)]


chosen_wt_AA = wt_AA[indices]
chosen_mut_AA = mut_AA[indices]
chosen_AA_positions = AA_positions[indices].astype(str)
chosen_fitness = fitness[indices]

mut_keys = chosen_wt_AA+chosen_AA_positions+chosen_mut_AA

# pdb.set_trace()

wt_seq = get_fasta_letters("inputs/beta_lac.fasta")


mut_seq_letter_list,mut_fitness_list = get_mut_letter_seq(wt_seq,chosen_fitness,mut_keys)


aaa = []
for i in range(len(mut_fitness_list)):
	aaa.append(str(mut_fitness_list[i]))

# pdb.set_trace()

# print(len(mut_fitness_list))
# print(len(mut_seq_letter_list))



# pdb.set_trace()
write_seq_text(mut_seq_letter_list,'beta_lac_seqs_big.txt')
# pdb.set_trace()
write_seq_text(aaa,'beta_lac_fitness_big.txt')



