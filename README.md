# resVAE ensemble

resVAE ensemble is the successor of [resVAE](https://github.com/lab-conrad/resvae) that utilizes rank aggregation with ensemble, among other improvements.

A very brief demo is shown here, please check it out: [Demo.ipynb](https://github.com/fwten/resVAE-ensemble/blob/main/Demo.ipynb)

The documentations will be updated shortly, in the meantime please check out the original [resVAE's notebook](https://github.com/lab-conrad/resVAE/blob/main/Example_notebook.ipynb) as well.

If you need any help or guidance, please do not hesitate to post in the [Discussions](https://github.com/lab-conrad/resVAE-ensemble/discussions) section.

## Getting started

### Prerequisites and installation

At the moment, it is recommended to create a new conda environment, clone this repository and then install the dependencies in this conda environment.

1. `conda create -n resvae python=3.10`
2. `conda activate resvae`
3. `git clone https://github.com/lab-conrad/resVAE-ensemble.git`
4. `cd resVAE-ensemble`
5. `pip install -r requirements.txt`

You should then be able to run the included `Demo.ipynb`.




## References and citations

If you find resVAE ensemble useful, please do cite us:

* Ten, F.W., Yuan, D., Jabareen, N. et al. Unsupervised identification of gene sets in multi-modal single-cell sequencing data using deep ensembles. Front. Cell Dev. Biol. 11:1091047 (2023). doi: 10.3389/fcell.2023.1091047

* Lukassen, S., Ten, F.W., Adam, L. et al. Gene set inference from single-cell sequencing data using a hybrid of matrix factorization and variational autoencoders. Nat Mach Intell 2, 800–809 (2020). doi: 10.1038/s42256-020-00269-9

