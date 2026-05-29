import sys
sys.path.append('/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen')

import yaml
import pandas as pd
from pathlib import Path
import muon as mu
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import anndata as ad
import scanpy as sc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from whole_mount_tumoroid.phenocoder.utils import load_plate
from whole_mount_tumoroid.phenocoder.cluster import run_clustering

from whole_mount_tumoroid.analysis.timecourse.run_mlp_training import ConditionalMarkerMLP


@torch.no_grad()
def predict_all_markers(
    model: ConditionalMarkerMLP,
    dapi_embeddings: torch.Tensor,
    num_markers: int,
    batch_size: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    model.eval().to(device)
    N = dapi_embeddings.shape[0]
    results = torch.zeros(N, num_markers)

    for marker_id in range(num_markers):
        for start in range(0, N, batch_size):
            end        = min(start + batch_size, N)
            batch      = dapi_embeddings[start:end].to(device)
            marker_ids = torch.full((end - start,), marker_id, dtype=torch.long, device=device)
            results[start:end, marker_id] = model(batch, marker_ids).cpu()

    return results 

def mlp_impute(mdata, mlp_dir, base_modality, imputations, models, n_comps_pca, resolution, stains):

    adata = mdata[base_modality]

    DAPI_DIM = adata.X.shape[1]
    NUM_MARKERS = len(stains)

    model = ConditionalMarkerMLP(
        dapi_dim=DAPI_DIM,
        num_markers=NUM_MARKERS,
        hidden_dims=[512, 256, 128],
        dropout=0.1,
    )

    for imputation in imputations:

        model_weights_path = f"{mlp_dir}/{models[imputation]}"
        print(f'Loading model {model_weights_path} for {imputation} imputation')

        model.load_state_dict(torch.load(model_weights_path))

        imputed = predict_all_markers(model, torch.tensor(adata.X, dtype=torch.float32), NUM_MARKERS, batch_size=128)

        imputed_df = pd.DataFrame(imputed, index=adata.obs_names, columns=stains)

        mdata.mod[f'{base_modality}_{imputation}_imputed'] = ad.AnnData(
            X=imputed_df.values,
            obs=adata.obs.copy(),  # carry over cell metadata
        )
        mdata.mod[f'{base_modality}_{imputation}_imputed'].var_names = stains

        mdata.mod[f'{base_modality}_{imputation}_imputed'] = run_clustering(mdata[f'{base_modality}_{imputation}_imputed'],
                                            n_comps=n_comps_pca,
                                            resolution=resolution,
                                            harmony=False,
                                            use_gpu=False,
                                            subsampling=True,
                                            frac=0.1,)

    return mdata

    