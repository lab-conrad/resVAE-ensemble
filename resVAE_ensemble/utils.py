#Copyright (C) 2019  Soeren Lukassen

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import json

import anndata as ad
import scanpy as sc
import dill

from tqdm import tqdm, trange


def one_hot_encoder(classes: np.ndarray, extra_dim_reserve=False, extra_dim_num=20):
    """
    A utility function to transform dense labels into sparse (one-hot encoded) ones. This wraps LabelEncoder from sklearn.preprocessing and to_categorical from keras.utils. If non-integer labels are supplied, the fitted LabelEncoder is returned as well.
    :param classes: A 1D numpy ndarray of length samples containing the individual samples class labels.
    :return: A 2D numpy ndarray of shape samples x classes and a None type if class labels were integers or a fitted instance of class sklearn.preprocessing.LabelEncder
    """
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder

    if extra_dim_reserve:
        classes_onehot, _ = one_hot_encoder(classes)
        classes_extra_dim = np.ones((classes_onehot.shape[0], extra_dim_num))
        classes_onehot_concat = np.concatenate((classes_extra_dim, classes_onehot), axis=1)
        return classes_onehot_concat, None
    else:
        if len(classes.shape) >= 2:
            individual_classes = np.apply_along_axis(one_hot_encoder, 0, classes)[0, 1:]
            classes_onehot_concat = np.concatenate(individual_classes, axis=1)
            return classes_onehot_concat, None
        else:
            if classes.dtype == 'O' or str(classes.dtype) == 'category' or str(classes.dtype)[0] == '<':
                l_enc = LabelEncoder()
                classes_encoded = l_enc.fit_transform(classes)
                classes_onehot = to_categorical(classes_encoded)
                return classes_onehot, l_enc
            elif classes.dtype in ['int16', 'int32', 'int64']:
                classes_onehot = to_categorical(classes)
                return classes_onehot, None
            else:
                raise ValueError('one hot encoder can not find numpy type')


def mixed_encoder(classes: np.ndarray, extra_dim_reserve=False, extra_dim_num=20, leave_out=None):
    """
    Mixed resVAE/VAE encoder: leave unrestricted dimensions to capture unknown effects.

    :param classes: np.ndarray with class assignments.
    :param extra_dim_reserve: Whether to reserve extra (unrestricted) dimensions (default: False)
    :param extra_dim_num: Number of extra dimensions (default: 20)
    :param leave_out: Whether to leave out a certain class (default: None)
    :return: returns a one-hot encoded class matrix
    """
    if leave_out is not None:
        classes = np.delete(classes, [0], axis=1)
        left_out = classes[:, leave_out].astype(dtype=np.int)
        classes = np.delete(classes, leave_out, axis=1)
    classes_onehot, _ = one_hot_encoder(classes, extra_dim_reserve, extra_dim_num)
    if leave_out is not None:
        classes_onehot = np.concatenate((classes_onehot, left_out), axis=1)
    return classes_onehot, None


def download_gmt(url: str, destination: str or None=None, file_name: str or None=None, replace: bool = False):
    """
    Utility function to download .gmt pathway files into the correct subfolder.

    :param url: The URL of the gmt files.
    :param destination: Destination directory. If left blank, defaults to 'gmts/'
    :param file_name: The file name to write to disk. If left blank, is left unchanged from the download source files name.
    :param replace: Boolean indicating whether to overwrite if the file name already exists.
    :return: None
    """
    import urllib3

    if destination is None:
        destination = 'gmts/'
    assert os.path.isdir(destination), print('Destination does not exist')
    if file_name is None:
        file_name = url.split('/')[-1]
    file = os.path.join(destination, file_name)
    if not replace:
        assert not os.path.exists(file), \
            print('File already exists and replace is set to False')
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    with open(file, 'wb') as out:
        while True:
            data = r.read()
            if not data:
                break
            out.write(data)
    r.release_conn()
    return None


def get_wikipathways(organism: str, destination: str or None=None, file_name: str or None=None, replace=False):
    """
    Utility function to quickly download WikiPathway data. A wrapper for download_gmt.

    :param organism: String (one of 'hs', 'mm', 'rn', 'dr', 'dm') to select download of the pathway data for either human, mouse, rat, zebra fish, or fruit fly.
    :param destination: Destination directory. If left blank, defaults to 'gmts/'
    :param file_name: The file name to write to disk. If left blank, is left unchanged from the download source files name.
    :param replace: Boolean indicating whether to overwrite if the file name already exists.
    :return: None
    """
    assert organism in ['hs', 'mm', 'rn', 'dr', 'dm'], print('Organism not found')
    if organism == 'hs':
        url = 'http://data.wikipathways.org/current/gmt/wikipathways-20190610-gmt-Homo_sapiens.gmt'
        download_gmt(url=url, destination=destination, file_name=file_name, replace=replace)
    return None

def gmt_to_df(infile: str) -> pd.DataFrame:
    """
    Convert .gmt files to pandas dataframe

    Args:
        infile (str): path to input .gmt file

    Returns:
        pd.DataFrame: .gmt annotations in dataframe form
    """

    gmt = []

    with open(infile, 'r') as f:
        for line in f:
            l = line.strip().split('\t', 2)
            if len(l) == 3:  # some pathways have no genes annotations
                gmt.extend(l)
            else:
                print(l)
    gmt = np.asarray(gmt, dtype=str)
    gmt = gmt.reshape((-1,3))

    gmt_pd = pd.DataFrame(gmt, columns=['TERM', 'DESC', 'GENES'])
    gmt_pd['GENES'] = gmt_pd['GENES'].replace('\\t', ';', regex=True)

    return gmt_pd


def gmt_to_json(infile: str, outfile: str or None=None):
    """
    Utility function to convert .gmt pathway files to json and write them to disk.

    :param infile: Path of the input file to convert.
    :param outfile: Output path of the corresponding .json
    :return: None
    """
    import codecs

    assert os.path.isfile(infile), print('Input file does not exist')
    if outfile is None:
        outfile = infile.split('.', 1)[0] + '.json'
    gmt = []
    with open(infile, 'r') as f:
        for line in f:
            gmt.append(line.strip().split('\t', 2))
    f.close()
    gmt = np.asarray(gmt)
    genes = []
    for line in gmt[:, 2].tolist():
        genes.append(line.split('\t'))
    genes = np.expand_dims(np.asarray(genes), axis=1)
    gmt = np.hstack([gmt[:, :2], genes]).tolist()
    json.dump(gmt, codecs.open(outfile, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


def read_json(infile: str):
    """
    Utility function to read a .json and report the result as a list.

    :param infile: Path to the input .json.
    :return: list of gne-pathway mappings
    """
    assert os.path.isfile(infile), print('Input file does not exist')
    with open(infile) as json_file:
        data: list = json.load(json_file)
    json_file.close()
    return data


def calculate_gmt_overlap(pathways, genelist):
    """
    Utility function to calculate the overlap between a gene set and any gene sets defined in a gmt file (nested list)

    :param pathways: A nested list such as would be obtained from running the read_json function
    :param genelist: Either a single list of genes to test, or a 2D numpy array with gene lists in columns
    :return: returns a numpy array with overlap counts
    """
    if max([len(x) for x in genelist[:][0]]) > 1:
        #genelist = np.asarray(genelist)
        hits = []
        for lst in range(len(genelist[0])):
            hits_int = []
            for i in range(len(pathways)):
                hits_int.append(len([x for x in pathways[i][2] if x in genelist[:, lst]]))
            hits.append(hits_int)
    else:
        hits = []
        for i in range(len(pathways)):
            hits.append(len([x for x in pathways[i][2] if x in genelist]))
        hits = np.asarray(hits)
    return hits


def calculate_fpc(weight_matrix: np.ndarray):
    """
    Utility function to calculate the fuzzy partition coefficient.

    :param weight_matrix: A numpy array with the neuron to gene mappings
    :return: The fuzzy partition coefficient of the weight matrix
    """
    n_genes = weight_matrix.shape[1]
    fpc = np.trace(weight_matrix.dot(weight_matrix.transpose())) / float(n_genes)
    return fpc


def normalize_count_matrix(exprs: np.ndarray):
    """
    Utility function to normalize a count matrix for the samples to sum to one. Wrapper for sklearn.preprocessing.Normalizer

    :param exprs: 2D numpy ndarray of shape samples x genes containing the gene expression values to be normalized to unit norm
    :return: a 2D numpy array of shape samples x genes of normalized expression values and a fitted instance of sklearn.preprocessing.Normalizer to be used for reconverting expression values after training resVAE
    """
    from sklearn.preprocessing import Normalizer

    norm = Normalizer(norm='l1', copy=False)
    norm_exprs = norm.fit_transform(exprs)
    return norm_exprs, norm


def load_sparse_matrix(sparse_matrix_path):
    """
    Utility function to load a sparse matrix.

    :param sparse_matrix_path: Path to matrix file
    :return: returns a np.ndarray
    """
    from scipy import io

    file_format = '.' + sparse_matrix_path.split('.')[-1]
    assert file_format == '.mtx'
    sparse_m = io.mmread(sparse_matrix_path)
    return sparse_m.toarray()


def write_sparse_matrix(sparse_matrix, sparse_matrix_path):
    """
    Utility function to write a sparse matrix.

    :param sparse_matrix: Sparse matrix to write to file.
    :param sparse_matrix_path: File name/path to write.
    :return: None
    """
    from scipy import io, sparse

    assert sparse_matrix.dtype == 'int64'
    sparse_m = sparse.csr_matrix(sparse_matrix)
    io.mmwrite(sparse_matrix_path, sparse_m)
    return None


def compose_dataframe(array, index, columns):
    """
    Utility function to convert a numpy array to a pandas DataFrame.

    :param array: the np.ndarray to convert
    :param index: row names
    :param columns: column names
    :return: a pandas DataFrame
    """
    assert len(array.shape) == 2
    assert array.shape[0] == len(index) and array.shape[1] == len(columns)
    return pd.DataFrame(array, index=index, columns=columns)


def decompose_dataframe(df):
    """
    Utility function to decompose a pandas DataFrame to numpy ndarrays.

    :param df: a pandas DataFrame
    :return: three np.ndarrays with the data, row names, and column names
    """
    index, column, array = df.index, df.column, df.values
    return array, index, column


def load_txt_file(txt_file_path):
    """
    Utility function to load a text file as list.

    :param txt_file_path: The path to the file (must be .txt)
    :return: The file content in list format
    """
    file_format = '.' + txt_file_path.split('.')[-1]
    assert file_format == '.txt'
    txt_file_array = np.loadtxt(txt_file_path, dtype=np.str)
    return txt_file_array.tolist()


def write_txt_file(txt_file_path, array):
    """
    Utility function to write a txt file.

    :param txt_file_path: The path and filename to write at
    :param array: The np.ndarray to write to file
    :return: None
    """
    np.savetxt(txt_file_path, array, delimiter=',')
    return None


def load_sparse(sparse_matrix_path, index_txt_file_path, column_txt_file_path):
    """
    Utility function to load a sparse matrix with row and column names as txt files.

    :param sparse_matrix_path: Path to sparse matrix
    :param index_txt_file_path: Path to row names
    :param column_txt_file_path: Path to column names
    :return: A DataFrame of a sparse matrix with row and column names
    """
    sparse_matrix = load_sparse_matrix(sparse_matrix_path)
    index, column = load_txt_file(index_txt_file_path), load_txt_file(column_txt_file_path)
    return compose_dataframe(sparse_matrix, index, column)


def write_sparse(df, sparse_matrix_path, index_txt_file_path, column_txt_file_path):
    """
    Utility function to write a DataFrame as sparse matrix, with row and column names.

    :param df: The DataFrame to write to file
    :param sparse_matrix_path: File path to write the sparse matrix to
    :param index_txt_file_path: row name file path
    :param column_txt_file_path: column name file path
    :return: None
    """
    sparse_matrix, index, column = decompose_dataframe(df)
    write_sparse_matrix(sparse_matrix, sparse_matrix_path)
    index = list(index)
    write_txt_file(index_txt_file_path, index)
    column = list(column)
    write_txt_file(column_txt_file_path, column)
    return None


def load_exprs(path, sep: str = ',', order: str = 'cg', sparse=None):
    """
    Utility function to load expression matrices, extract gene names, and return both

    Currently supports 'csv', 'tsv', 'pkl', 'feather' files

    :param path: Path to the expression matrix
    :param sep: Separator used in the expression file (default: ',')
    :return: a 2D numpy ndarray with the expression values and a pandas index object containing the ordered gene names
    """
    if sparse:
        sparse_matrix_path, index_txt_file_path, column_txt_file_path = (sparse['sparse_matrix_path'],
                                                                         sparse['index_txt_file_path'],
                                                                         sparse['column_txt_file_path'])
        exprs = load_sparse(sparse_matrix_path, index_txt_file_path, column_txt_file_path)
    else:
        # TODO: Update Docstring
        assert os.path.isfile(path), print('Invalid file path')
        assert order in ['cg', 'gc']
        ext = path.split('.')[-1]
        assert ext in ['csv', 'tsv', 'pkl', 'feather'], print('Unrecognized file format. Currently supported formats include: csv, tsv, pkl and feather.')
        if ext == 'csv':
            exprs = pd.read_csv(path, sep=sep, header=0, index_col=0)
        elif ext == 'tsv':
            exprs = pd.read_csv(path, sep='\t', header=0, index_col=0)
        elif ext == 'pkl':
            exprs = pd.read_pickle(path)
        elif ext == 'feather':
            exprs = pd.read_feather(path)
        if not order == 'cg':
            exprs = exprs.T
    return np.asarray(exprs), exprs.columns


def read_h5ad(path, keep_raw=True, verbose=True):
    """
    Loads an AnnData HDF5 file
    """
    import anndata as ad
    h5ad = ad.read_h5ad(path)
    if not keep_raw:
        del h5ad.raw

    if verbose:
        print(h5ad)
        if h5ad.__dict__.get('_X') is not None:
            print(f"X.min() = {h5ad.__dict__.get('_X').min()}")
            print(f"X.max() = {h5ad.__dict__.get('_X').max()}")
        else:
            print('.X absent!')
        if h5ad.__dict__.get('_raw') and h5ad.__dict__.get('_raw').__dict__.get('_X') is not None:
            print(f".raw.X.min() = {h5ad.__dict__.get('_raw').__dict__.get('_X').min()}")
            print(f".raw.X.max() = {h5ad.__dict__.get('_raw').__dict__.get('_X').max()}")
        else:
            print('.raw.X absent!')
    return h5ad

def train_ensemble(counts,
                    encodings,
                    labels,
                    genes,
                    config,
                    n_runs,
                    model_name,
                    model_dir):
    
    from tensorflow.keras import backend as K
    from .resvae import resVAE

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for i in range(n_runs):
        try:
            del(resvae_model)
            gc.collect()
        except:
            pass

        K.clear_session()
        resvae_model = resVAE(model_dir=model_dir, config=config)
        resvae_model.genes = genes
        resvae_model.add_latent_labels(encodings)
        assert config['INPUT_SHAPE'][0] == len(resvae_model.genes)
        assert config['INPUT_SHAPE'][1] == len(labels)
        resvae_model.compile()

        with open(f'{model_dir}/{model_name}_run_{i}_labels.txt', 'w') as f:
            print(*[x for x in labels], sep='\n', file=f)
        
        hist, _ = resvae_model.fit(exprs=counts, classes=encodings,
                                  model_dir=model_dir, model_name=f'{model_name}_run_{i}', verbose=0)
        
        print('Saving model...')
        resvae_model.save_model_new(model_dir, model_name=f'{model_name}_run_{i}', only_resvae=True)

        print(f'\n@==============================@ Finished run # {i}', model_name)


def load_multirun(mname: str, in_dir: str = 'data/models', out_dir: str = 'data/outputs', genes=None,
                  nrun: int = 0, weights_key: str = 'weights_clusters', use_dill: str = None, as_df: bool = True, dump_dill: bool = False):
    """Load multiple resVAE runs' weights

    Args:
        mname (str): name of the model folder/file
        in_dir (str, optional): path to find the model folders. Defaults to 'data/models'.
        out_dir (str, optional): path to find the output folders associated with the model. Defaults to 'data/outputs'.
        genes ([type], optional): either path to txt file with genes list, or np.array of gene names. Defaults to None.
        nrun (int, optional): number of runs to load. Defaults to 20.
        weights_key (str, optional): get the specified weights. Defaults to latent_to_gene's `weight_clusters`.
        use_dill (str, optional): check and use dill file if present. Defaults to None.
        as_df (bool, optional): return as pd.DataFrame or dict. Defaults to True.

    Returns:
        pd.DataFrame: df of the
    """
    import tensorflow as tf
    from .resvae import resVAE

    in_dir = f'{in_dir}/{mname}'
    out_dir = f'{out_dir}/{mname}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    multiruns = dict()

    if use_dill and os.path.exists(f'{out_dir}/{use_dill}'):
        multiruns = load_dill(f'{out_dir}/{use_dill}')
        print(f'Dill file loaded: {out_dir}/{use_dill}')
        print('Set `use_dill` to `False` to load from resVAE weights!!')

    else:
        print(f'Dill file not found: {out_dir}/{use_dill}')
        with open(f'{in_dir}/{mname}_run_0.config') as d:
            config = json.load(d)

        if isinstance(genes, str):
            genes = np.loadtxt(genes, dtype=str, delimiter='\t')
        elif isinstance(genes, np.ndarray):
            pass
        else:
            print('Please make sure to provide proper gene names!')

        #K.clear_session()
        with tf.device('/cpu:0'):
            resvae_model = resVAE(model_dir=in_dir, config=config)
            resvae_model.genes = genes

            assert len(genes) == int(config["INPUT_SHAPE"][0]), "Incorrect number of genes!!"

            resvae_model.compile()

            if not nrun:
                nrun = sum([x.endswith('.config') for x in os.listdir(f'{in_dir}') ])

            for i in trange(nrun):

                l_encoder_all = np.loadtxt(f'{in_dir}/{mname}_run_{i}_labels.txt', dtype=str, delimiter='\t')
                resvae_model.add_latent_labels(l_encoder_all)
                resvae_model.load_model_new(in_dir, model_name=f'{mname}_run_{i}', only_resvae=True)

                if weights_key == 'weights_clusters':
                    multiruns.update({i: {
                        'weights_clusters': resvae_model.get_latent_to_gene(normalized=True)
                    } })
                elif weights_key == 'weights_clusters_nonnorm':
                    multiruns.update({i: {
                        'weights_clusters_nonnorm': resvae_model.get_latent_to_gene(normalized=False)
                    } })
                elif weights_key == 'weights_neurons_1':
                    multiruns.update({i: {
                        'weights_neurons_1': resvae_model.get_neuron_to_gene(normalized=True, initial_layer=1),
                    } })
                elif weights_key == 'weights_neurons_2':
                    multiruns.update({i: {
                        'weights_neurons_2': resvae_model.get_neuron_to_gene(normalized=True, initial_layer=2),
                    } })
                elif weights_key == 'weights_latent_neurons_1':
                    multiruns.update({i: {
                        'weights_latent_neurons_1': resvae_model.get_latent_to_neuron(normalized=True, target_layer=1),
                    } })
                elif weights_key == 'weights_latent_neurons_2':
                    multiruns.update({i: {
                        'weights_latent_neurons_2': resvae_model.get_latent_to_neuron(normalized=True, target_layer=2),
                    } })
                elif weights_key == 'biases':
                    multiruns.update({i: {
                        'biases': resvae_model.get_gene_biases(relative=False),
                    } })
                elif weights_key == 'biases_relative':
                    multiruns.update({i: {
                        'biases_relative': resvae_model.get_gene_biases(relative=True)
                    } })
                else:
                    print('Key error for weights extraction!!')
                    return

        if dump_dill:
            print('Writing .dill and h5ad files...')
            with open(f"{out_dir}/compare10runs_dict.dill", "wb") as dill_file:
                dill.dump(multiruns, dill_file)

    # df manipulation part
    if as_df:
        multiruns_df = dict_to_df(multiruns, key=weights_key)
        if max(multiruns_df.shape) != len(resvae_model.genes) and 'biases' not in weights_key:
            multiruns_df['_label'] = multiruns_df.index.values
            return multiruns_df

        elif 'biases' not in weights_key:
            multiruns_df.columns = np.concatenate((resvae_model.genes, ['_run']))
            multiruns_df['_label'] = multiruns_df.index.values
            ad_compare10runs = ad.AnnData(X=multiruns_df)
            ad_compare10runs.write_h5ad(f'{out_dir}/ad_compare10runs_{weights_key}.h5ad')

        else:
            multiruns_df['gene'] = np.tile(genes, len(multiruns.keys()))

        return multiruns_df
    else:
        return multiruns

def load_dill(path):
    with open(f"{path}", "rb") as dill_file:
        return dill.load(dill_file)

def write_dill(obj, path):
    with open(f"{path}", "wb") as dill_file:
        dill.dump(obj, dill_file)

def dict_to_df(d, key='weights_clusters'):
    """
    Takes a dict of the extracted weights from multiple resVAE runs
    and turn it info a wide dataframe where columns are genes
    """
    df = None
    df_temp = None
    for i in d.keys():
        df_temp = pd.DataFrame(d[i][key]).reset_index()
        df_temp['_run'] = i

        if i == 0:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp])

    df['_run'] = df['_run'].astype(str).astype('category')
    df['index'] = df['index'].astype(str)

    return df.set_index('index')


def calculate_elbow(weights: np.ndarray, negative: bool = False):
    """
    Calculates the position of the elbow point for each weight matrix.

    :param weights: A 1D or 2D numpy ndarray of length genes or shape neurons x genes containing weight mappings
    :param negative: Boolean indicating whether to return negatively enriched indices (default: False)
    :return: Returns an integer (1D input) or 1D numpy ndarray (2D input) with the position of the elbow point along a sorted axis
    """
    if weights.ndim == 1:
        if negative:
            weights_current = np.sort(np.abs(weights[weights < 0] / np.min(weights[weights < 0])))
            weights_index = np.arange(len(weights_current)) / np.max(len(weights_current))
            distance = weights_index - weights_current
            return len(weights_index) - np.argmax(distance)
        else:
            weights_current = np.sort(np.abs(weights[weights >= 0] / np.max(weights[weights >= 0])))
            weights_index = np.arange(len(weights_current)) / np.max(len(weights_current))
            distance = weights_index - weights_current
            return np.argmax(distance) + np.sum(weights < 0)
    if weights.ndim > 1:
        distances = []
        if negative:
            weights_index = np.arange(weights.shape[1]) / np.min(weights.shape[1])
            for neuron in range(weights.shape[0]):
                weights_current = np.sort(np.abs(weights[neuron, weights < 0] / np.max(weights[neuron, weights < 0])))
                distance = weights_index - weights_current
                distances.append(len(weights_index) - np.argmax(distance))
        else:
            weights_index = np.arange(weights.shape[1]) / np.max(weights.shape[1])
            for neuron in range(weights.shape[0]):
                weights_current = np.sort(np.abs(weights[neuron, weights < 0] / np.max(weights[neuron, weights < 0])))
                distance = weights_index - weights_current
                distances.append(np.argmax(distance) + np.sum(weights < 0))
        return distances

def calculate_elbow_fw2(weights, bins=5, verbose=True):
    """
    In this elbow calculation method, we sort the data array and chop it into N pieces:

        |-----|-----|-----|-----|-----|
           1     2     3     4     5

    The knee point is usually at the 1st, and the elbow at the 5th.

    To find the cut-off point, we basically take all the data in 1st or 5th section,
    then draw a line connecting the lowest/highest point to the median (can be mean too?) of the section data.

    Then, we rotate it such that the connecting line becomes the horizontal y-axis,
    and the max or min point as the cut-off.

    weights: weights_clusters
    bins: (int) number of bins or pieces

    """
    def find_elbow(weight_column, theta):
        data = np.array([np.arange(len(weight_column)), np.sort(weight_column)]).T

        # make rotation matrix
        cos = np.cos(theta)
        sin = np.sin(theta)
        rotation_matrix = np.array(((cos, -sin), (sin, cos)))

        # rotate data vector
        rotated_vector = data.dot(rotation_matrix)

        # return index of elbow
        return np.where(rotated_vector == rotated_vector.min())[0][0], rotated_vector

    def get_data_radian_top(weight_column):
        data = np.array([np.arange(len(weight_column)), np.sort(weight_column)]).T
        data = data[len(weight_column) - len(weight_column)//bins:]

        return np.arctan2(data[:, 1].max() - np.median(data[:, 1]),
                          data[:, 0].max() - np.median(data[:, 0]))

    def get_data_radian_bot(weight_column):
        data = np.array([np.arange(len(weight_column)), np.sort(weight_column)]).T
        data = data[:len(weight_column)//bins]
        return np.arctan2(np.median(data[:, 1]) - data[:, 1].min(),
                          np.median(data[:, 0]) - data[:, 0].min())

    def get_knees_elbows(weights):
        weights = np.sort(weights)

        # find top first
        _, rot_dat_top = find_elbow(weights, get_data_radian_top(weights))
        rot_dat_df_top = pd.DataFrame(rot_dat_top)
        rot_dat_df_top.columns = ['x', 'y']
        poscut = rot_dat_df_top.query(f'y == {rot_dat_top[:, 1].min()}').index.values[0]
        # found the index of this guy, actual index in whole data is thus:
        #poscut = weights.shape[0] - poscut - 1

        # find bot next
        _, rot_dat_bot = find_elbow(weights, get_data_radian_bot(weights))
        rot_dat_df_bot = pd.DataFrame(rot_dat_bot)
        rot_dat_df_bot.columns = ['x', 'y']
        negcut = rot_dat_df_bot.query(f'y == {rot_dat_bot[:, 1].max()}').index.values[0]
        # negative, so count from bottom
        negcut = negcut

        return negcut, poscut, rot_dat_df_top, rot_dat_df_bot

    return get_knees_elbows(weights)

def compare_cutoff(weights_clusters, bins=0, verbose=False):
    """
    Plots the ranked weights and the cut-off points using different methods.

    eg: cutils.compare_cutoff(compare10runs[0]['weights_clusters'].loc[cluster], bins=10)

    weights_clusters: weights for one cluster

    returns: figure and cut-off values

    """
    plt.figure(figsize=(12, 3))

    if bins:
        segment = len(weights_clusters)//bins
        plt.vlines([ (x+1)*segment for x in range(bins)],
                    ymin=np.min(weights_clusters)-.2,
                    ymax=np.max(weights_clusters)+.2,
                linestyles='solid', color='crimson', linewidth=1, alpha=0.7)

    pos_cutoff = calculate_elbow(weights_clusters)
    neg_cutoff = calculate_elbow(weights_clusters, negative=True)
    plt.gcf().subplots_adjust(bottom=0.35)
    plt.plot(np.sort(weights_clusters), color='steelblue')
    plt.vlines([pos_cutoff, neg_cutoff],
                ymin=np.min(weights_clusters)-.2,
                ymax=np.max(weights_clusters)+.2,
                linestyles='solid', color='darkorange', linewidth=2, alpha=1)
    plt.title(f'Gene weight cutoffs - {weights_clusters.name}')
    plt.xlabel('Gene rank')
    plt.ylabel('Weight')
    plt.tight_layout()

    # fw2
    fw2_neg_cutoff, fw2_pos_cutoff, _, _ = calculate_elbow_fw2(weights_clusters, bins=bins, verbose=verbose)
    plt.vlines([fw2_neg_cutoff, fw2_pos_cutoff],
                ymin=np.min(weights_clusters)-.4,
                ymax=np.max(weights_clusters)+.4,
              linestyles='dotted', color='lime', linewidth=2, alpha=0.9)

    if verbose:
        print('SL: ', pos_cutoff, neg_cutoff)
        print('FW 2: ', fw2_pos_cutoff, fw2_neg_cutoff)

    return pos_cutoff, neg_cutoff, fw2_pos_cutoff, fw2_neg_cutoff


def plot_cutoffs(df, label, method="SL", bins=0, drop=['_run', '_label'], color=None, verbose=False):
    """
    Plots the ranked weights and cut-offs over all runs

    df: pd.DataFrame containing the weights
    label: (str) cluster label pattern (filter rows by pattern)
    method: (str) elbow calculation method; currently 'SL' or 'FW'
    
    color: (str) use only this color if specified

    returns: figure of the cut-offs over all runs

    """
    plt.figure(figsize=(12, 6))
    N = len(df['_run'].unique())

    for r in range(N):
        _df = df.loc[label].drop(drop, axis=1)

        if method == "FW2":
            neg_cutoff, pos_cutoff, _, _ = calculate_elbow_fw2(_df.iloc[r, :], bins=bins, verbose=False)
        else:
            pos_cutoff = calculate_elbow(_df.iloc[r, :])
            neg_cutoff = calculate_elbow(_df.iloc[r, :], negative=True)

        if verbose:
            print(f'{pos_cutoff=} {neg_cutoff=}')

        plt.gcf().subplots_adjust(bottom=0.35)

        if color:
            plt.plot(np.sort(_df.iloc[r, :]), color=color, alpha=0.3, label=r)
        else: # rainbow
            plt.plot(np.sort(_df.iloc[r, :]), color=f'C{r}', alpha=0.5, label=r)
            
        plt.vlines([pos_cutoff],
                    ymin=np.min(_df.iloc[r, :])-.2,
                    ymax=np.max(_df.iloc[r, :])+.2,
                    linestyles='--', color=f'red', alpha=0.4)
        plt.vlines([neg_cutoff],
                    ymin=np.min(_df.iloc[r, :])-.2,
                    ymax=np.max(_df.iloc[r, :])+.2,
                    linestyles='--', color=f'blue', alpha=0.4)
        plt.title(f'Gene weight cutoffs - {label} {N} runs')
        plt.xlabel('Gene rank')
        plt.ylabel('Weight')
        plt.tight_layout()

def aggregate_cutoffs(df, classes, nrun=0, method='SL', bins=5):
    """
    Aggregates the resVAE cutoffs for each cluster across multiple runs

    TODO: currently only positives, maybe add negatives?

    df: (pd.DataFrame) containing the resVAE weight matrices (labels as rows x genes as columns)
    classes: (list) list of clusters/labels to be included

    returns: (dict) of clusters/labels and their respective cutoffs
    """
    cluster_cutoffs = {}

    if not nrun:
        nrun = len(df.index) // len(classes)

    for cond in tqdm(classes):
        df_ = df.loc[cond].drop(['_run', '_label'], axis=1)
        c_cutoff = []
        for i in range(nrun):
            if method == "FW2":
                neg_cutoff, pos_cutoff, _, _ = calculate_elbow_fw2(df_.iloc[i, :], bins=bins)
            else:
                pos_cutoff = calculate_elbow(df_.iloc[i, :])
                neg_cutoff = calculate_elbow(df_.iloc[i, :], negative=True)

            c_cutoff.append(df_.shape[1] - pos_cutoff)

        cluster_cutoffs.update({cond: int(np.median(c_cutoff))})

    return cluster_cutoffs

def aggregate_cutoffs_neg(df, classes, nrun=0, method='SL', bins=5):
    """
    Aggregates the resVAE cutoffs for each cluster across multiple runs

    TODO: currently only positives, maybe add negatives?

    df: (pd.DataFrame) containing the resVAE weight matrices (labels as rows x genes as columns)
    classes: (list) list of clusters/labels to be included

    returns: (dict) of clusters/labels and their respective cutoffs
    """
    cluster_cutoffs = {}

    if not nrun:
        nrun = len(df.index) // len(classes)

    for cond in tqdm(classes):
        df_ = df.loc[cond].drop(['_run', '_label'], axis=1)
        #g_freq = []
        c_cutoff = []
        for i in range(nrun):
            if method == "FW2":
                neg_cutoff, pos_cutoff, _, _ = calculate_elbow_fw2(df_.iloc[i, :], bins=bins)
            else:
                neg_cutoff = calculate_elbow(df_.iloc[i, :])

            if neg_cutoff > df_.shape[1]//2:
                print(f'Something might be wrong: {neg_cutoff=} {i=} {cond}')

            if pos_cutoff > neg_cutoff:
                pos_cutoff, neg_cutoff = neg_cutoff, pos_cutoff
                print(f'Swapped cutoffs: {pos_cutoff=} {neg_cutoff=} {i=} {cond}')

            c_cutoff.append(neg_cutoff)

        cluster_cutoffs.update({cond: int(np.median(c_cutoff))})

    return cluster_cutoffs

def aggregate_rankings(df, classes, nrun=0):
    """
    Aggregates the ranked genes for each cluster across every runs

    df: (pd.DataFrame) containing the resVAE weight matrices (labels as rows x genes as columns)
    classes: (list) list of clusters/labels to be included

    returns: (dict) of (pd.DataFrame) containing the ranked genes of all runs for each cluster
    """
    clusters_ranking = {}

    if not nrun:
        nrun = len(df.index) // len(classes)

    for cond in tqdm(classes):
        meta_df = pd.DataFrame(columns=[f'run_{a}' for a in range(nrun)])
        df_ = df.loc[cond].drop(['_run', '_label'], axis=1)

        for i in range(nrun):
            iter_genes = df_.iloc[i,:].sort_values(ascending=False).index
            meta_df[f'run_{i}'] = iter_genes

        clusters_ranking.update({cond: meta_df})

    return clusters_ranking

def rra_clusters_export(rra_clusters):
    """
    In one long table, list all the genes that made the cut-off for all clusters
    """
    for i, cluster in enumerate(rra_clusters.keys()):
        if not i:
            rra_export = rra_clusters[cluster].copy(deep=True)
            rra_export['cluster'] = cluster
        else:
            _df = rra_clusters[cluster].copy(deep=True)
            _df['cluster'] = cluster
            rra_export = pd.concat([rra_export, _df])
    
    return rra_export

def rra_ranking_consistency(clusters_ranking, k=-1, n=None):
    """
    Investigate how consistently ranked the features are:
    find who the top K features are, and how often they show up in top N features

    clusters_ranking: the rra ranked gene lists
    """
    from operator import itemgetter

    df = pd.DataFrame(
        columns=["cluster", "top_features", "counts", "k_sum", "sum", "overlap", "k"]
        )
    for cluster in clusters_ranking.keys():
        top_list = list(zip(*np.unique(clusters_ranking[cluster].iloc[:k], return_counts=True)))
        top_list.sort(key=itemgetter(1), reverse=True)
        features, counts = zip(*top_list)
        
        top_k_features = features[:k]
        sum_k_counts = sum(counts[:k])

        overlap = sum_k_counts / sum(counts)

        df.loc[cluster] = [cluster, top_k_features, counts[:k], sum_k_counts, sum(counts), overlap, k]

    return df

def get_labels_pvals(d, sort_by='Name', which='BH_adj', index='_label'):
    """
    Collects the scores or p-vals from a dictionary for each label

    d: (dict) containing the scores or p-vals and labels
    sort_by: (str) column name to sort
    which: (str) column name to collect
    index: (str) column name for encoder labels

    returns: pd.DataFrame of p-vals with labels as rows and genes as columns

    """
    for x in d.keys():
        d[x] = d[x].sort_values(sort_by)

    df = None
    df_temp = None
    for i,k in enumerate(d.keys()):
        df_temp = d[k][[which]].reset_index()

        if i == 0:
            df = df_temp.copy()
            df.rename(columns = {which: k}, inplace=True)
        else:
            df[k] = d[k][[which]].values

    df = df.set_index('index')
    df.index.name = index
    return df.T

def add_encoders(metadata, exclude=None, include=None,
                 free=None):
    """
    Add labels to encoder for CAVE, e.g.: 'M', 'F', 'smoker', 'healthy', etc

    metadata: pd.DataFrame containing the metadata
    exclude: list of column names to exclude
    include: list of column names to include
    free: number of free dimensions

    returns: np.array of one-hot encoded categorical clusters, encoded classes

    """
    categorical_clusters = []
    encoder_classes = []
    if exclude and len(exclude) > 0:
        metadata = metadata.drop(exclude, axis=1)
    elif include and len(include) > 0:
        metadata = metadata.loc[:,include]
    else:
        return 'Please specify the labels that you want to encode with either `exclude` or `include`!'

    for col in metadata.select_dtypes(include=[int]).columns:
        metadata.loc[:,col] = metadata.loc[:,col].apply(lambda x: f'{col}_{x}')

    metadata = metadata.astype(str).astype('category')

    for col in metadata.columns:
        encoded, encoder = one_hot_encoder(metadata.loc[:,col])
        categorical_clusters.append(encoded)
        encoder_classes.append(encoder.classes_)
        print(encoded.shape, encoder.classes_.shape)

    categorical_clusters = np.hstack(categorical_clusters)
    encoder_classes = np.concatenate(encoder_classes)

    if free:
        categorical_clusters, encoder_classes = add_free_labels(categorical_clusters, encoder_classes, n=free)

    return categorical_clusters, encoder_classes

def add_free_labels(categorical_clusters, encoder_classes, n):
    """
    Add free dimensions to categorical clusters and encoded labels

    categorical_clusters: np.array of one-hot encoded categorical clusters
    encoder_classes: np.array of the encoded labels
    n: number of free dimensions to add
    """
    return np.hstack([categorical_clusters, np.ones((categorical_clusters.shape[0], n))]), np.append(encoder_classes, [f'free{x}' for x in range(n)])

def smooth_labels(y, smooth_factor=0.1, axis=1):
    """Converts a matrix of one-hot encoded labels into smoothed version

    Args:
        y (matrix): Matrix of one-hot encoded labels
        smooth_factor (float, optional): Smoothing factor. Defaults to 0.1.
        axis (int, optional): Smoothing axis. Defaults to 1.

    Raises:
        Exception: Invalid smoothing factor, should be between 0 and 1

    Returns:
        y (matrix): Matrix of smoothed labels
    """
    assert len(y.shape) == 2

    if axis == 1:
        y = y.T

    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    if axis == 1:
        y = y.T

    return y


def assert_config(config: dict):
    assert len(config['INPUT_SHAPE']) == 2, print('Input shape has wrong dimensionality')
    assert type(config['ENCODER_SHAPE']) == list, \
        print('Encoder shape is not a list')
    assert len(config['ENCODER_SHAPE']) >= 1, \
        print('Missing encoder dimensions')
    assert type(config['DECODER_SHAPE']) == list,\
        print('Decoder shape is not a list')
    assert len(config['DECODER_SHAPE']) >= 1, \
        print('Missing decoder dimensions')
    assert type(config['DROPOUT']) in [int, type(None), float], print('Invalid value for dropout')
    if config['DROPOUT']:
        assert config['DROPOUT'] < 1, print('Dropout too high')
    assert type(config['LATENT_SCALE']) == int and config['LATENT_SCALE'] >= 1, \
        print('Invalid value for latent scale. Please choose an integer larger than or equal to one')
    assert type(config['BATCH_SIZE']) == int and config['BATCH_SIZE'] >= 1, \
        print('Invalid value for batch size. Please choose an integer larger than or equal to one')
    assert type(config['EPOCHS']) == int and config['EPOCHS'] >= 1, \
        print('Invalid value for epochs. Please choose an integer larger than or equal to one')
    assert type(config['STEPS_PER_EPOCH']) in [type(None), int], \
        print('Invalid value for steps per epoch. Please choose None or an integer larger than or equal to one')
    assert type(config['VALIDATION_SPLIT']) in [float, type(None)], \
        print('Invalid value for validation split. Please choose None or a float value smaller than one')
    assert type(config['LATENT_OFFSET']) in [float, int], \
        print('Please choose a number for the latent offset')
    assert config['DECODER_BIAS'] in ['last', 'all', 'none'], \
        print('Invalid value for decoder bias. Please choose all, none, or last')
    assert config['DECODER_REGULARIZER'] in ['none', 'l1', 'l2', 'l1_l2',
                                             'var_l1', 'var_l2', 'var_l1_l2', 'dot', 'dot_weights'], \
        print('Invalid value for decoder regularizer. Please choose one of '
              'none, l1, l2, l1_l2, var_l1, var_l2, or var_l1_l2')
    if config['DECODER_REGULARIZER'] != 'none':
        assert type(config['DECODER_REGULARIZER_INITIAL']) == float, \
            print('Please choose a float value as (initial) decoder regularizer penalty')
    assert config['BASE_LOSS'] in ['mse', 'mae'], \
        print('Please choose mse or mae as base loss')
    assert type(config['DECODER_BN']) == bool, \
        print('Please choose True or False for the decoder batch normalization')
    assert type(config['CB_LR_USE']) == bool, \
        print('Please choose True or False for the learning rate reduction on plateau')
    assert type(config['CB_ES_USE']) == bool, \
        print('Please choose True or False for the early stopping callback')
    if config['CB_LR_USE']:
        assert type(config['CB_LR_FACTOR']) == float, \
            print('Please choose a decimal value for the learning rate reduction factor')
        assert type(config['CB_LR_PATIENCE']) == int and config['CB_LR_PATIENCE'] >= 1, \
            print('Please choose an integer value equal to or larger than one for the learning rate reduction patience')
        assert type(config['CB_LR_MIN_DELTA']) == float or config['CB_LR_MIN_DELTA'] == 0, \
            print('Please choose a floating point value or 0 for the learning rate reduction minimum delta')
    if config['CB_ES_USE']:
        assert type(config['CB_ES_PATIENCE']) == int and config['CB_ES_PATIENCE'] >= 1, \
            print('Please choose an integer value equal to or larger than one for the early stopping patience')
        assert type(config['CB_ES_MIN_DELTA']) == float or config['CB_ES_MIN_DELTA'] == 0, \
            print('Please choose a floating point value or 0 for the early stopping minimum delta')
    if config['CB_LR_USE'] or config['CB_ES_USE']:
        assert config['CB_MONITOR'] in ['loss', 'val_loss'], \
            print('Please choose loss or val_loss as metric to monitor for callbacks')
    assert type(config['MULTI_GPU']) in ['bool', 'int']

def is_dense(obj):
    return isinstance(obj, (np.ndarray, np.matrix, ad._core.views.ArrayView))

def is_csr(obj):
    from scipy import sparse
    return isinstance(obj, sparse.csr.csr_matrix)

def check_and_normalize_counts(obj, plot=True):
    """Check and normalize the counts matrix

    Args:
        obj (matrix-like): Counts matrix
        plot (bool, optional): Plot a visual summary. Defaults to True.

    Returns:
        norm_counts (np.array): Normalized counts matrix
        normalizer (sklearn normalizer): sklearn normalizer
    """
    if is_csr(obj):
        print('is sparse csr!')
        norm_counts, normalizer = normalize_count_matrix(obj.todense())
        if plot:
            sns.heatmap(obj.todense()[::200,::200], cmap='turbo')
    elif is_dense(obj):
        print('is dense!')
        norm_counts, normalizer = normalize_count_matrix(obj)
        if plot:
            sns.heatmap(obj[::200,::200], cmap='turbo')
    else:
        norm_counts = None
        print(f'Check your count matrix type!: {type(obj)}')

    assert isinstance(norm_counts, np.ndarray)
    return norm_counts, normalizer

def remove_zero_genes(anndata, min_cells=3, verbose=True):
    """Remove zero genes

    Args:
        anndata (ad.AnnData): AnnData object
        min_cells (int, optional): Minimum number of cells. Defaults to 3.
        verbose (bool, optional): Print removed genes. Defaults to True.

    Returns:
        _type_: _description_
    """
    genes_to_keep = anndata.var_names[sc.pp.filter_genes(anndata, inplace=False, min_cells=min_cells)[0]]
    if verbose:
        pl(anndata.var_names[~sc.pp.filter_genes(anndata, inplace=False, min_cells=min_cells)[0]])
    return anndata[:,genes_to_keep]

def check_zero_genes(anndata, df=None, min_cells=3):
    """Check for zero genes

    Args:
        anndata (ad.AnnData): AnnData object
        df (pd.DataFrame, optional): Pandas DataFrame. Defaults to None.
        min_cells (int, optional): Minimum number of cells to be present in. Defaults to 3.

    Returns:
        df (pd.DataFrame, optional): Matrix of the filtered features
    """
    import anndata as ad

    genes_to_filter = anndata.var_names[~sc.pp.filter_genes(anndata, inplace=False, min_cells=min_cells)[0]]

    if genes_to_filter.any():
        from scipy import sparse

        if isinstance(anndata[:,genes_to_filter].X, (np.matrix, ad._core.views.ArrayView)):
            if anndata[:,genes_to_filter].X.max():
                sns.heatmap(anndata[:,genes_to_filter].X[:,::20], cmap='turbo')
                plt.title('anndata matrix of these filtered genes')
                plt.show()
            print(anndata[:,genes_to_filter].X.max())
        elif isinstance(anndata[:,genes_to_filter].X, sparse.csr.csr_matrix):
            if anndata[:,genes_to_filter].X.todense().max():
                sns.heatmap(anndata[:,genes_to_filter].X.todense()[:,::20], cmap='turbo')
                plt.title('anndata (from sparse) matrix of these filtered genes')
                plt.show()
            print(anndata[:,genes_to_filter].X.todense().max())
        else:
            print('Check your matrix type, doesn\'t seem to be np.matrix or csr.csr_matrix')
            print(f'Instead we found: {type(anndata[:,genes_to_filter].X)}')

        if not isinstance(df, type(None)):
            sns.heatmap(df.loc[:, genes_to_filter], cmap='turbo')
            plt.title('resvae matrix of these filtered genes')

            return df.loc[:, genes_to_filter]
    else:
        print('No zero-genes found!')

#import collections
from itertools import chain, repeat, islice

def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)

def pl(list_like):
    print(*sorted(list_like), sep='\n')

def pw(list_like):
    print(*sorted(list_like), sep=' ')
