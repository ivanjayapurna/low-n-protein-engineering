# Evotuning 

1. InterPro (PFAM) Protein Sequence Fetch
Code snippet(s) to fetch protein sequences of proteins in the dienelactose hydrolase (DLH) family (as well as lipase and cutinase) - chosen based on InterPro and PFAM sequence matches to wild-type isPETase. There are ~70k entries in DLH then an additional ~20k in lipase and cutinase.
2. Run generate_evotune_inputs.ipynb to generate train, in_domain validation and out_domain validation scripts for evotuning in the format required for UniRep.
3. Run evotune.py to evotune [TODO: incude (out_set) validation error for stopping condition & dump weights every few thousand iterations in case of a crash or increase in validation error]

Note: filter_top_100.ipynb is legacy script used when I was attempting to use jackHMMer - decided not to go forward with it.
Note 2: NOT using tf_evotune.py - instead finished a JAX implementation of evotuning which will be used instead. 