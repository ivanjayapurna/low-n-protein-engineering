# Low-N Protein Engineering
This repository contains a complete, open-source, end-to-end re-implementation of the Church Lab's eUniRep in silico protein engineering pipeline as presented in Biswas et al. Details on our implementation can be read here [INSERT LINK TO PUBLISHED AUTHOREA], the original Church lab paper can be read here: [INSERT LINK TO LOW-N PAPER], with their repository here: [INSERT TO THEIR REPOSITORY]. The JAX-unirep re-implementation we use in our implementation can be found here: [INSERT LINK TO JAX-UNIREP].

## How to use:
Each in silico step in the protein engineering pipeline has a jupyter notebook that will execute that step as well as an individual README file. The pipeline steps have been broken down as follows:

1. Training UniRep: either use the weights provided by the Church lab [LINK THEIR REPO AGAIN] or use JAX-unirep reimplementation to re-train from scratch [LINK TO JAX-UNIREP], both are well documented for this step.
2. Curating pre-training set for evotuning
3. Evotuning: we pushed an example script to the jax-unirep repo [LINK IT AGAIN]
4. Top model selection and hyperparamter tuning
5. Markov Chain Monte Carlo (MCMC) directed evolution
6. Additional: scripts to do further analysis such as PCA and epistasis evaluation

If you want to request any modifications / additions or want to collaborate feel free to start an issue / PR!
