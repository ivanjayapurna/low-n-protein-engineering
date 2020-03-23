from Bio import SeqIO



####################
###### INPUTS ######
####################

# input FASTA sequence
in_fasta = 'A0A0K8P6T7'

# input mutation codes AS STRING:
mutations_string = 'S214H-I168R-W159H-S188Q-R280A-A180I-G165A-Q119Y-L117F-T140D'
mutations_list = ['S214H', 'I168R', 'W159H']

# in and out paths
in_path = 'inputs/' + in_fasta + '.fasta.txt'
out_path = 'outputs/' + mutations_string + '_' + in_fasta + '.fasta.txt'

####################
#### FUNCTIONS #####
####################

def mutate(sequence, mutations, list_or_string='string'):
    if (list_or_string == 'string'):
        mutations = mutations.split('-')

    mut_sequence = list(sequence)
    for mut in mutations:
        curr_res = mut[0]
        mut_res = mut[-1]
        mut_pos = mut[1:-1]
        mut_sequence[int(mut_pos)-1] = mut_res

    return ''.join(mut_sequence)


####################
###### SCRIPT ######
####################

fasta_seqs = SeqIO.parse(open(in_path),'fasta')
out_file = open(out_path, "w")
for fasta in fasta_seqs:
    name, sequence = fasta.id, str(fasta.seq)
    out_file.write(mutations_string + '_' + name + '\n')
    mut_seq = mutate(sequence,mutations_string, 'string')
    out_file.write(mut_seq)

out_file.close()