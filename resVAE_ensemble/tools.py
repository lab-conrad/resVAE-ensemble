import sys
import re
import json
import requests
from time import sleep

import numpy as np
import pandas as pd

def get_enrichr_results(gene_set_library, genelist, description):
    """Query Enrichr

    Args:
        gene_set_library (str): Enrichr gene set library
        genelist (list): List of genes to query
        description (str, optional): Description of the genes

    Raises:
        Exception: Response error

    Returns:
        dict: Enrichr results
    """
    ADDLIST_URL = 'https://maayanlab.cloud/Enrichr/addList'
    payload = {
        'list': (None, genelist),
        'description': (None, description)
    }

    response = requests.post(ADDLIST_URL, files=payload)
    if not response.ok:
        raise Exception('Error analyzing gene list')
    sleep(1)
    data = json.loads(response.text)

    RESULTS_URL = 'https://maayanlab.cloud/Enrichr/enrich'
    query_string = '?userListId=%s&backgroundType=%s'
    user_list_id = data['userListId']
    response = requests.get(RESULTS_URL + query_string % (user_list_id, gene_set_library))
    sleep(1)
    #return [data['shortId'], json.loads(response.text)]
    return json.loads(response.text)

def get_libraries():
    """List available Enrichr libraries

    Returns:
        libs (list): Available Enrichr libraries
    """
    libs_json = json.loads(requests.get('https://maayanlab.cloud/Enrichr/datasetStatistics').text)
    libs = [lib['libraryName']for lib in libs_json['statistics']]
    return libs

def query_enrichr(library, genelist, clean=False):
    """Enrich gene lists using Enrichr

    Args:
        library (list): Enrichr library names
        genelist (list): Query gene list
        clean (bool, optional): Clean up the gene names by removing decimal part. Defaults to False.

    Returns:
        data (pd.DataFrame): Enrichment results table
    """
    if clean:
        genelist = [re.sub(r'\.\d*$',r'',x) for x in genelist]
    genelist = '\n'.join(genelist)
    data = get_enrichr_results(library, genelist, 'Sample gene list')
    #print('https://maayanlab.cloud/enrich?dataset={0}'.format(data[0]))
    return data #None

def get_enrichr_df(library:str, genelist:list, clean:bool=True, go_id:bool=True):
    """
    Query EnrichR and return a dataframe!
    
    :param library: (str) name of query library
    :param genelist: (list) list of genes to query
    :param clean: (bool) clean gene names by removing decimal dots and numbers
    :param go_id: (bool) extract GO:ID as its own column in dataframe   
    :return: (pd.DataFrame) enrichment results
    """
    if library in get_libraries():
        df = query_enrichr(library, genelist, clean)
        df = pd.DataFrame.from_dict(df[library])
        try:
            df.columns = ['Rank', 'Term name', 'P-value', 'Z-score', 'Combined score', 'Overlapping genes', 'Adjusted p-value', 'Old p-value', 'Old adjusted p-value']
            if go_id:
                df['GO'] = df['Term name'].str.extract('.*\((GO\:.*)\)$', expand=True)
        except:
            print(f'{library}\n{genelist}')
            print(f'Might be empty, attention!')
            return df
        return df
    else:
        return f'{library} is not a valid EnrichR library!'


def q_check(rra_df:pd.DataFrame, library:str='GO_Biological_Process_2021'):
    """Quickly check for enriched terms from RRA results

    Args:
        rra_df (pd.DataFrame): RRA dataframe
        library (str, optional): Enrichr library. Defaults to 'GO_Biological_Process_2021'.

    Returns:
        df (pd.DataFrame): Enrichr results
    """
    genes = rra_df.query('BH_adj < 0.05').Name.values
    if len(genes):
        df = get_enrichr_df(library, genes, clean=True, go_id=True)
    else:
        return 'No genes detected in this cluster!'
    return df

def qenr(genes:list, library:str='GO_Biological_Process_2021'):
    """Quick query of genes

    Args:
        genes (list): Genes
        library (str, optional): Enrichr library. Defaults to 'GO_Biological_Process_2021'.

    Returns:
        df (pd.DataFrame): Enrichr results
    """
    if len(genes):
        df = get_enrichr_df(library, genes, clean=True, go_id=True)
    else:
        return 'No genes detected in this cluster!'
    return df

def rra_rankMatrix(df):
    """
    Return the ranking number for each feature across the diff. run as a dataframe.
    
    df: (pd.DataFrame) gene x run matrix
    
    """    
    ranked_clusters_ranking = pd.DataFrame()
    for i, _ in enumerate(df.columns):
        if not i:
            ranked_clusters_ranking['Name'] = df[f'run_{i}'].sort_values().values
        ranked_clusters_ranking[f'run_{i}'] = df[f'run_{i}'].sort_values().index

    ranked_clusters_ranking = ranked_clusters_ranking.set_index('Name')
    ranked_clusters_ranking += 1
    ranked_clusters_ranking /= len(ranked_clusters_ranking)
    return ranked_clusters_ranking.sort_values(by='run_0')

def rra_betaScores(ranks):
    """RRA's betaScores
    """
    from scipy.stats import beta
    if isinstance(ranks, pd.Series):
        ranks = ranks.values
    n = np.isfinite(ranks).sum()
    beta_distr1 = beta(np.arange(1,n+1), np.flip(np.arange(1, n+1), axis=0))
    betas = beta_distr1.cdf(np.sort(ranks))
    return(betas)

def rra_correctBetaPvalues(betas):
    """RRA's correctBetaPvalues
    """
    k = np.isfinite(betas).sum()
    p = np.minimum(np.min(betas * k), 1)
    return p

def rra_rhoScores(genes):
    """Calculate the rho

    Args:
        genes (list): list of the genes

    Returns:
        rho: corrected betas
    """
    rho = rra_betaScores(genes)
    rho = rra_correctBetaPvalues(rho)
    return rho

def pyRRA(df):
    """Perform RRA on all runs

    Args:
        df (pd.DataFrame): DataFrame of genes x runs

    Returns:
        df (pd.DataFrame): Consensus DataFrame
    """
    from statsmodels.stats.multitest import fdrcorrection
    
    df = rra_rankMatrix(df)
    df = pd.DataFrame(df.apply(rra_rhoScores, axis=1), columns=['Score']).reset_index()
    df['BH_adj'] = fdrcorrection(df['Score'])[1]
    df['Name_cleaned'] = df['Name'].str.replace(r'\.\d*$', r'', regex=True)
    df = df.sort_values(by='BH_adj')
    return df
    
def pyRRA_mp(clusters_ranking, workers, cutoffs=None, dill=False):
    """parallel RRA

    Args:
        clusters_ranking (_type_): ranked clusters
        workers (int): number of workers
        cutoffs (np.array, optional): cut-off values for the clusters. Defaults to None.
        dill (bool, optional): save as dill. Defaults to False.

    Returns:
        pyRRA_all: RRA results
    """
    import mpire

    def do_pyRRA(cluster):
        df = pyRRA(clusters_ranking[cluster])
        return cluster, df
    
    pyRRA_all = {}
    
    with mpire.WorkerPool(n_jobs=workers, use_dill=dill) as pool:
        for result in pool.imap_unordered(do_pyRRA, clusters_ranking.keys(), progress_bar=True):
            pyRRA_all.update({
                result[0]: result[1][:cutoffs[result[0]]] if cutoffs else result[1]
            })
    
    return pyRRA_all


def test_maic(maic_df):
    """Ranking aggregation using Meta-analysis information content

    ## running MAIC
    # python maic.py -f ./RRA_cluster_transposed.csv -o maic_out -v

    # the input .csv file should look like this: (tab separated)
    # resvae0 run0 RANKED NAMED_GENES gene1 gene2 gene3 ...
    # resvae1 run1 RANKED NAMED_GENES gene1 gene2 gene3 ...
    # resvae2 run2 RANKED NAMED_GENES gene1 gene2 gene3 ...

    Args:
        maic_df (df): df with the ranking of the genes from each resvae run
    """
    sys.path.append('maic/')
    from maic import maic2

    app = maic2.Maic2()
    app.run(maic_df)
