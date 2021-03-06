# Pre-Evotuning
There are two (near identical) scripts in this folder that walk you through curating a pre-training dataset on the order of 10,000's of evolutionarily related sequences to your target protein. Detailed instructions are described at the top of each jupyter notebook. The difference in the two scripts is one is set-up for IsPETase which had an abundance of similar sequences (thus the curation process was fairly simple), in comparison to the MS2 protein which required a bit more effort and human decision making i.e. method variability.

In general these methods follow the following steps:
1. Search up the wild-type protein sequence on InterPro and PFAM Protein Sequence Fetch to get what family / clan it is in. Download all these sequences as well as other related families / clans as suggested by PFAM. We recommend downloading via code snippets that fetch protein sequences using the InterPro API (these can be generated by InterPro).
2. If you have less sequences than desired (i.e. <50,000 for example, as note that some of these sequences may be duplicates or be invalid sequences) then search for keywords on InterPro and download those as outlined in the 2MS2 script.
3. Run the generate_evotune_inputs jupyter notebook.
4. Follow the example evotune.py script on the jax-unirep github!
