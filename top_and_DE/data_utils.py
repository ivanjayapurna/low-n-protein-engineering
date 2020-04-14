"""
Utilities for data processing.
"""
import numpy as np
import os
"""
File formatting note.
Data should be preprocessed as a sequence of comma-seperated ints with
sequences  /n seperated
"""

# Lookup tables
aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10,
    'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24,
    'stop':25,
}

int_to_aa = {value:key for key, value in aa_to_int.items()}

def get_aa_to_int():
    """
    Get the lookup table (for easy import)
    """
    return aa_to_int

def get_int_to_aa():
    """
    Get the lookup table (for easy import)
    """
    return int_to_aa

# Helper functions

def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]

def int_seq_to_aa(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return "".join([int_to_aa[i] for i in s])

def aas_to_int_seq(aa_seq):
    int_seq = ""
    for aa in aa_seq:
        int_seq += str(aa_to_int[aa]) + ","
    return str(aa_to_int['start']) + "," + int_seq + str(aa_to_int['stop'])

# Preprocessing in python
def fasta_to_input_format(source, destination):    
    sourcefile = os.path.join(source)
    destination = os.path.join(destination)
    with open(sourcefile, 'r') as f:
        with open(destination, 'w') as dest:
            seq = ""
            for line in f:
                if line[0] == '>' and not seq == "":
                    dest.write(aas_to_int_seq(seq) + '\n')
                    seq = ""
                elif not line[0] == '>':
                    seq += line.replace("\n","")
