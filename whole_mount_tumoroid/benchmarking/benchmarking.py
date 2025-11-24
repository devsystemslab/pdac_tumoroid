from whole_mount_tumoroid.benchmarking.mlami import compute_mlami
from whole_mount_tumoroid.benchmarking.nasw import compute_nasw
from whole_mount_tumoroid.benchmarking.gcs import compute_gcs
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import scanpy as sc
from harmonypy.lisi import compute_lisi
import numpy as np
import pandas as pd
import sys
import os
from contextlib import contextmanager
from tqdm import tqdm
import warnings
from pathlib import Path

def run_benchmark_sample(adata, sample_id, n_neighbors, benchmarks, latent_key):
    """
    Run benchmarking on a single sample
    :param adata:
    :param sample_id:
    :param n_neighbors:
    :param benchmarks:
    :return:
    """
    assert latent_key in ['X','pca']

    df = pd.DataFrame({'id':sample_id}, index=[0])
    adata = adata[adata.obs['id'] == sample_id].copy()

    if 'cnmi' in benchmarks:
        df['cnmi_phenocoder'] = normalized_mutual_info_score(adata.obs['leiden'], adata.obs['leiden_phenocoder'])
        df['cnmi_phenocoder_msg'] = normalized_mutual_info_score(adata.obs['leiden'],
                                                                 adata.obs['leiden_phenocoder_msg'])
        df['cnmi_nuclei'] = normalized_mutual_info_score(adata.obs['leiden'], adata.obs['leiden_nuclei'])
        df['cnmi_nuclei_msg'] = normalized_mutual_info_score(adata.obs['leiden'], adata.obs['leiden_nuclei_msg'])

    if 'ari' in benchmarks:
        df['ari_phenocoder'] = adjusted_rand_score(adata.obs['leiden'], adata.obs['leiden_phenocoder'])
        df['ari_phenocoder_msg'] = adjusted_rand_score(adata.obs['leiden'], adata.obs['leiden_phenocoder_msg'])
        df['ari_nuclei'] = adjusted_rand_score(adata.obs['leiden'], adata.obs['leiden_nuclei'])
        df['ari_nuclei_msg'] = adjusted_rand_score(adata.obs['leiden'], adata.obs['leiden_nuclei_msg'])
        df['ari_imputed_nuclei'] = adjusted_rand_score(adata.obs['leiden'], adata.obs['leiden_imputed_nuclei'])
        df['ari_imputed_neighbors'] = adjusted_rand_score(adata.obs['leiden'], adata.obs['leiden_imputed_neighbors'])

    if any(b in benchmarks for b in ['gcs', 'mlami', 'nasw', 'clisis']):
        # set up spatial
        adata.obsm['spatial'] = adata.obs[['centroid-0', 'centroid-1', 'z']].values.copy()
        if latent_key=='pca':
            # run pca
            sc.pp.pca(adata, n_comps=int(adata.X.shape[1]/2))
            # neighbors for pca space and spatial space
            sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=n_neighbors, key_added=latent_key)
            adata.uns[f'{latent_key}_n_neighbors'] = n_neighbors
        else:
            # neighbors for pca space and spatial space
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=n_neighbors, key_added=latent_key)
            adata.uns[f'{latent_key}_n_neighbors'] = n_neighbors

        sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=n_neighbors, key_added='spatial')

    if 'gcs' in benchmarks:
        df['gcs'] = compute_gcs(adata, spatial_knng_key='spatial',latent_knng_key=latent_key)

    if 'clisis' in benchmarks:
        try:
            if latent_key=='pca':
                cell_lisis_latent = compute_lisi(adata.obsm['X_pca'], adata.obs, label_colnames=['leiden'])
            else:
                cell_lisis_latent = compute_lisi(adata.X, adata.obs, label_colnames=['leiden'])
            cell_lisis_spatial = compute_lisi(adata.obsm['spatial'], adata.obs, label_colnames=['leiden'])
            cell_clisis = cell_lisis_latent / cell_lisis_spatial
            cell_log_clisis = np.log2(cell_clisis)
            n_cell_types = adata.obs['leiden'].nunique()
            max_cell_log_clisis = np.log2(n_cell_types / 1)
            norm_log_clisis = cell_log_clisis / max_cell_log_clisis
            df['clisis'] = (1 - np.nanmedian(abs(norm_log_clisis)))
        except Exception as e:
            df['clisis'] = np.nan

    if 'mlami' in benchmarks:
        df['mlami'] = compute_mlami(adata,spatial_knng_key='spatial',latent_knng_key=latent_key, n_neighbors=n_neighbors, res_num=5)

    if 'nasw' in benchmarks:
        if latent_key=='pca':
            df['nasw'] = compute_nasw(adata,latent_knng_key='pca',latent_key='X_pca',n_neighbors=n_neighbors, res_num=5)
        else:
            adata.obsm[latent_key] = adata.X.copy()
            df['nasw'] = compute_nasw(adata,latent_knng_key=latent_key,latent_key='X',n_neighbors=n_neighbors, res_num=5)

    return df

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def run_benchmark_dataset(mdata, sample_ids, batch_id, out_tmp:str, n_neighbors=15, benchmarks=None, save=True):
    """
    Run benchmarking on a dataset
    :param mdata:
    :param batch:
    :param batch_size:
    :param out_tmp:
    :param n_neighbors:
    :param benchmarks:
    :param save:
    :return:
    """
    if benchmarks is None:
        benchmarks = ['gcs', 'clisis', 'nasw', 'cnmi','ari','mlami']
    warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

    # Validate modalities
    mod_keys = ['phenocoder', 'phenocoder_msg', 'nuclei', 'nuclei_msg', 'imputed_nuclei_bytimepoints_False', 'imputed_neighbors_bytimepoints_False']
    for key in mod_keys:
        assert key in mdata.mod.keys()

    # Setup observations
    for mod_key in mod_keys:
        mdata.mod[mod_key].obs['leiden_nuclei'] = mdata.mod['nuclei'].obs['leiden'].copy()
        mdata.mod[mod_key].obs['leiden_nuclei_msg'] = mdata.mod['nuclei_msg'].obs['leiden'].copy()
        mdata.mod[mod_key].obs['leiden_phenocoder'] = mdata.mod['phenocoder'].obs['leiden'].copy()
        mdata.mod[mod_key].obs['leiden_phenocoder_msg'] = mdata.mod['phenocoder_msg'].obs['leiden'].copy()
        mdata.mod[mod_key].obs['leiden_imputed_nuclei'] = mdata.mod['imputed_nuclei_bytimepoints_False'].obs['leiden'].copy()
        mdata.mod[mod_key].obs['leiden_imputed_neighbors'] = mdata.mod['imputed_neighbors_bytimepoints_False'].obs['leiden'].copy()
        
        if 'id' not in mdata.mod[mod_key].obs.columns:
            mdata.mod[mod_key].obs['id'] = mdata.mod[mod_key].obs['plate_id'].astype(str) + '_' + mdata.mod[mod_key].obs[
            'well_id'].astype(str)

    results = []
    for sample_id in tqdm(sample_ids, desc='Running Benchmarks'):
        try:
            df_nuclei = run_benchmark_sample(mdata.mod['nuclei'], sample_id, n_neighbors, benchmarks, latent_key='pca')
            df_nuclei_msg = run_benchmark_sample(mdata.mod['nuclei_msg'], sample_id, n_neighbors, benchmarks, latent_key='pca')
            df_phenocoder = run_benchmark_sample(mdata.mod['phenocoder'], sample_id, n_neighbors, benchmarks, latent_key='X')
            df_phenocoder_msg = run_benchmark_sample(mdata.mod['phenocoder_msg'], sample_id, n_neighbors, benchmarks, latent_key='X')
            df_imputed_nuclei = run_benchmark_sample(mdata.mod['imputed_nuclei_bytimepoints_False'], sample_id, n_neighbors, benchmarks, latent_key='X')
            df_imputed_neighbors = run_benchmark_sample(mdata.mod['imputed_neighbors_bytimepoints_False'], sample_id, n_neighbors, benchmarks, latent_key='X')
            
            df_sample = pd.concat([df_phenocoder, df_nuclei, df_phenocoder_msg, df_nuclei_msg, df_imputed_nuclei, df_imputed_neighbors],
                            keys=['phenocoder', 'nuclei', 'phenocoder_msg', 'nuclei_msg', 'imputed_nuclei', 'imputed_msg'])
            results.append(df_sample)
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            continue
    df_results = pd.concat(results).reset_index()
    df_results = df_results.rename(columns={'level_0': 'type'})
    df_results.drop(['level_1'], axis=1, inplace=True)

    # write results to csv
    if save:
        df_results.to_csv(Path(out_tmp,f'benchmark_batch_{batch_id}.csv'), index=False)
    else:
        return df_results
