import mudata
import scanpy as sc
import anndata as ad
import bbknn
import matplotlib.pyplot as plt
import rapids_singlecell as rsc
from tqdm import tqdm
import pandas as pd
import numpy as np
from whole_mount_tumoroid.phenocoder.spatial import get_chulls_connected_components
import muon as mu

def run_umap_gpu(mdata, rep='X'):
    """
    Run UMAP on GPU for all modalities in mdata
    :param mdata:
    :param rep:
    :return:
    """
    for mod in tqdm(mdata.mod_names, desc='Running UMAP'):
        adata = mdata[mod].copy()
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.neighbors(adata, use_rep=rep)
        rsc.tl.umap(adata, n_components=2)
        rsc.get.anndata_to_CPU(adata)
        mdata.mod[mod] = adata.copy()
    mdata.update()
    return mdata

def add_chull_stats_to_mdata_org(mdata, df_chulls_agg, batch_correction=True,mod='phenocoder_combined', select_variable_features=True):
    """
    Add chull data to X of selected mod in mdata.
    :param mdata:
    :param mod:
    :param df_chulls_agg:
    :param batch_correction:
    :param select_variable_features:
    :return:
    """
    adata_init = mdata[mod].copy()
    df_chulls_agg.index = df_chulls_agg['well_id'] + '_' + df_chulls_agg['plate_id']
    # arrange by adata.obs.index
    df_chulls_agg = df_chulls_agg.loc[adata_init.obs.index]
    adata = ad.AnnData(X=np.concatenate([adata_init.layers['raw'], df_chulls_agg.drop(['plate_id','well_id'], axis=1).to_numpy()], axis=1))
    adata.var_names = list(adata_init.var_names) + list(df_chulls_agg.columns.drop(['plate_id','well_id']))
    adata.obs = adata_init.obs.copy()
    adata.layers['raw'] = adata.X.copy()
    sc.pp.scale(adata)
    adata.X[np.isnan(adata.X)] = 0
    if select_variable_features:
        sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, n_comps=32)
    if batch_correction:
        bbknn.bbknn(adata, batch_key='plate_id')
    else:
        sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.leiden(adata, resolution=1)
    sc.tl.umap(adata, n_components=2, min_dist=0.5)
    mdata.mod[mod] = adata.copy()
    mdata.update()
    return mdata

def merge_org_embeddings(mdata_source, mdata_target, mod='phenocoder_combined', batch_correction=True, select_variable_features=True):
    """
    Merge organoid embeddings
    :param mdata_source:
    :param mdata_target:
    :param mod:
    :param batch_correction:
    :param select_variable_features:
    :return:
    """
    # select modality
    adata_source = mdata_source[mod].copy()
    adata_target = mdata_target[mod].copy()
    # get common labels
    labels_target = adata_target.obs.index.values
    labels_source = adata_source.obs.index.values
    labels = list(set(labels_target).intersection(set(labels_source)))
    idx_target = adata_target.obs.index[adata_target.obs.index.isin(labels)]
    idx_source = adata_source.obs.index[adata_source.obs.index.isin(labels)]
    adata_target = adata_target[idx_target]
    adata_source = adata_source[idx_source]
    # construct merged adata
    df = adata_target.obs.merge(adata_source.obs['leiden'], left_index=True, right_index=True,
                                suffixes=('_target', '_source'))
    adata = ad.AnnData(X=np.concatenate([adata_target.layers['raw'], adata_source.layers['raw']], axis=1))
    adata.obs = df
    var_names_target = [name + '_target' for name in adata_target.var_names]
    var_names_source = [name + '_source' for name in adata_source.var_names]
    adata.var_names = var_names_target + var_names_source
    adata.layers['raw'] = adata.X.copy()
    sc.pp.scale(adata)
    adata.X[np.isnan(adata.X)] = 0
    if select_variable_features:
        sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, n_comps=32)
    if batch_correction:
        bbknn.bbknn(adata, batch_key='plate_id')
    else:
        sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.leiden(adata, resolution=1)
    sc.tl.umap(adata, n_components=2, min_dist=0.5)
    return adata

def add_features_to_obs(mdata, source, target, layer=None):
    """
    Add features to obs
    :param mdata:
    :param source:
    :param target:
    :param layer:
    :return:
    """
    if layer is not None:
        df = pd.DataFrame(mdata[source].layers[layer], columns=mdata[source].var_names, index=mdata[source].obs.index)
    else:
        df = pd.DataFrame(mdata[source].X, columns=mdata[source].var_names, index=mdata[source].obs.index)
    mdata.mod[target].obs = pd.concat([mdata[target].obs, df], axis=1)
    mdata.update()
    return mdata

def run_chulls_connected_components(mdata: mu.MuData, clusters: list[str], mod: str = 'phenocoder') -> pd.DataFrame:
    """
    Run convex hulls for individual connected components for all samples in dataset for given modality.
    :param mdata:
    :param clusters:
    :param mod:
    :return:
    """
    df_iter = mdata.mod[mod].obs.groupby(['well_id', 'plate_id'], observed=True).size().reset_index()
    df_chulls = []
    for well, plate in tqdm(zip(df_iter['well_id'], df_iter['plate_id']), total=df_iter.shape[0],
                            desc='Computing convex hulls'):
        df_chulls_sample = get_chulls_connected_components(mdata.mod[mod], well, plate, clusters)
        df_chulls.append(df_chulls_sample)
    df_chulls = pd.concat(df_chulls).reset_index(drop=True)
    # group by well and plate -> get mean and sum for all other columns
    agg_dict = {col: ['mean', 'sum'] for col in df_chulls.columns if col not in ['well_id', 'plate_id']}
    df_chulls_agg = df_chulls.groupby(['well_id', 'plate_id']).agg(agg_dict)
    # reset row and column multi index
    df_chulls_agg.columns = ['_'.join(col).strip() for col in df_chulls_agg.columns.values]
    # add number of connected components -> len df_chulls
    df_chulls_agg['n_chulls'] = df_chulls.shape[0]
    df_chulls_agg.reset_index(inplace=True)
    # left merge with df_iter and fill missing values matches with zeros
    df_chulls_agg = df_iter.loc[:,['well_id','plate_id']].merge(df_chulls_agg, on=['well_id', 'plate_id'], how='left')
    df_chulls_agg.fillna(0, inplace=True)
    return df_chulls, df_chulls_agg
