# resVAE ensemble

resVAE ensemble is the successor of [resVAE](https://github.com/lab-conrad/resvae) that utilizes rank aggregation with ensemble, among other improvements.

Give it a try if you:

1. Have feature-barcode matrix data (scRNA-seq counts, scATAC-seq peaks, output from CellRanger etc)
2. Have some sort of annotations for each barcode/cells (cell type, cluster ID, sample ID, pathology, phenotype, treatment, ...)
3. Want to extract gene sets that characterize the given annotation(s), e.g.: lists of genes that characterize phenotype A, B, C
4. Want to use these genes for further analyses or investigations, especially if you are already using Seurat, Scanpy, Signac, ...

## Background

**✨ [Publication](https://www.frontiersin.org/articles/10.3389/fcell.2023.1091047/full)**

Feature identification and manual inspection is currently still an integral part of biological data analysis in single-cell sequencing. Features such as expressed genes and open chromatin status are selectively studied in specific contexts, cell states or experimental conditions. While conventional analysis methods construct a relatively static view on gene candidates, artificial neural networks have been used to model their interactions after hierarchical gene regulatory networks. However, it is challenging to identify consistent features in this modeling process due to the inherently stochastic nature of these methods. Therefore, we propose using ensembles of autoencoders and subsequent rank aggregation to extract consensus features in a less biased manner. Here, we performed sequencing data analyses of different modalities either independently or simultaneously as well as with other analysis tools. Our method can successfully complement and find additional unbiased biological insights with minimal data processing or feature selection steps while giving a measurement of confidence, especially for models using stochastic or approximation algorithms. In addition, our method can also work with overlapping clustering identity assignment suitable for transitionary cell types or cell fates in comparison to most conventional tools.


## Getting started

A very brief demo is shown here, please check it out: [Demo.ipynb](https://github.com/fwten/resVAE-ensemble/blob/main/Demo.ipynb)

The documentations will be updated shortly, in the meantime please check out the original [resVAE's notebook](https://github.com/lab-conrad/resVAE/blob/main/Example_notebook.ipynb) as well.

If you need any help or guidance, please do not hesitate to post in the [Discussions](https://github.com/lab-conrad/resVAE-ensemble/discussions) section.


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

