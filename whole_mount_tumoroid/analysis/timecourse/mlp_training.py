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


def get_train_val_test_indices(n_samples, val_frac=0.15, test_frac=0.15, random_state=42):
    indices = np.arange(n_samples)
    
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    
    n_test = int(n_samples * test_frac)
    n_val  = int(n_samples * val_frac)
    
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    
    return train_idx, val_idx, test_idx

def regress_out_z(df: pd.DataFrame,
                  intensity_col: str = "intensity_value",
                  stain_col: str = "stain",
                  z_col: str = "z") -> pd.DataFrame:
    df = df.copy()

    def _regress(group):
        X = group[[z_col]].values          # shape (n, 1)
        y = group[intensity_col].values    # shape (n,)
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        group = group.copy()
        group[intensity_col] = residuals
        return group

    df = df.groupby(stain_col, group_keys=False).apply(_regress)
    return df

def scale_per_stain(df: pd.DataFrame,
                    intensity_col: str = "intensity_value",
                    stain_col: str = "stain",
                    zero_center: bool = True,
                    max_value: float | None = None) -> pd.DataFrame:
    df = df.copy()
    scaled = df.groupby(stain_col)[intensity_col].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-10) if zero_center
                  else x / (x.std(ddof=1) + 1e-10)
    )
    if max_value is not None:
        scaled = scaled.clip(-max_value, max_value)
    df[intensity_col] = scaled
    return df

def data_setup(adata, df_stain_layout, df, imputation, stain_dict):
    obs = adata.obs.copy()
    obs = obs.reset_index().merge(df_stain_layout, how='left', on=['well_id', 'plate_id'])
    obs = obs.merge(df[['label',f'ch_02_{imputation}',f'ch_03_{imputation}',f'ch_04_{imputation}']], how='left', on=['label'])
    
    obs_df = pd.melt(obs, id_vars=['label', 'staining_set', 'z'], value_vars=[f'ch_02_{imputation}', f'ch_03_{imputation}', f'ch_04_{imputation}'], var_name='channel', value_name='intensity_value')
    obs_df['channel'] = obs_df['channel'].str.split('_').str[1].astype(int)
    obs_df['stain'] = obs_df.apply(
        lambda row: stain_dict.get(str(row['staining_set']), {}).get(row['channel']), 
        axis=1
    )
    obs_df['stain'] = pd.Categorical(obs_df['stain'])
    obs_df['stain_id'] = obs_df['stain'].cat.codes

    obs_df = regress_out_z(obs_df)
    obs_df = scale_per_stain(obs_df)

    X = adata.X.copy()
    X_long = np.tile(X, (3,1))

    embeddings = torch.tensor(X_long, dtype=torch.float32)
    marker_ids = torch.tensor(obs_df['stain_id'].values)
    intensities = torch.tensor(obs_df['intensity_value'].values, dtype=torch.float32)

    return embeddings, marker_ids, intensities, obs_df

def dataloader_from_indices(embeddings, marker_ids, intensities, indices, batch_size=256, shuffle=True):
    ds = MarkerDataset(embeddings[indices], marker_ids[indices], intensities[indices])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class MarkerDataset(Dataset):
    """
    Each sample = (dapi_embedding, marker_id, intensity_value)
    
    Expects:
        embeddings : np.ndarray  [N, emb_dim]  — one per cell
        marker_ids : np.ndarray  [N]            — integer marker index
        intensities: np.ndarray  [N]            — float target
    """
    def __init__(self, embeddings, marker_ids, intensities):
        self.embeddings  = torch.tensor(embeddings,  dtype=torch.float32)
        self.marker_ids  = torch.tensor(marker_ids,  dtype=torch.long)
        self.intensities = torch.tensor(intensities, dtype=torch.float32)

    def __len__(self):
        return len(self.intensities)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.marker_ids[idx], self.intensities[idx]

class ConditionalMarkerMLP(nn.Module):
    """
    Predicts signal intensity for a given (DAPI embedding, marker) pair.

    Architecture:
        - Concatenate DAPI embedding + marker embedding
        - MLP with residual connections
        - Scalar output
    """
    def __init__(
        self,
        dapi_dim: int,
        num_markers: int,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_markers = num_markers

        in_dim = dapi_dim + num_markers

        # Build MLP trunk
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Optional residual projection (if in_dim != last hidden)
        self.residual_proj = (
            nn.Linear(in_dim, hidden_dims[-1])
            if in_dim != hidden_dims[-1]
            else nn.Identity()
        )

        self.head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, dapi_emb: torch.Tensor, marker_ids: torch.Tensor) -> torch.Tensor:
        one_hot = torch.zeros(marker_ids.shape[0], self.num_markers, device=dapi_emb.device)
        one_hot.scatter_(1, marker_ids.unsqueeze(1), 1.0)

        x = torch.cat([dapi_emb, one_hot], dim=-1)
        out = self.mlp(x) + self.residual_proj(x)
        return self.head(out).squeeze(-1)

def train(
    model: ConditionalMarkerMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "runs/marker_mlp",
):
    print(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_loss = 0.0
        for dapi, marker_ids, targets in train_loader:
            dapi, marker_ids, targets = dapi.to(device), marker_ids.to(device), targets.to(device)
            preds = model(dapi, marker_ids)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            global_step += 1

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dapi, marker_ids, targets in val_loader:
                dapi, marker_ids, targets = dapi.to(device), marker_ids.to(device), targets.to(device)
                val_loss += loss_fn(model(dapi, marker_ids), targets).item()

        scheduler.step()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalars("Loss/epoch", {"train": avg_train, "val": avg_val}, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
            writer.add_histogram(f"Weights/{name}", param.data, epoch)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train MSE: {train_loss/len(train_loader):.4f} | "
            f"Val MSE:   {val_loss/len(val_loader):.4f}"
        )

    return model

def train_mlp(params, mdata, mlp_dir, base_modality, imputations, epochs, stains, stain_dict, df_plate_layouts):
    dir_screen = params['dir_screen']
    plates = params['plates']

    df_stain_layout = df_plate_layouts[['well_id', 'plate_id', 'staining_set']]
    df_stain_layout = df_stain_layout.map(str)

    #get channel columns
    df_plates = []
    for plate in plates:
        df_plate = load_plate(plate, params['input_type'], dir_screen, params['registered'], z_step=10, plate_id=f'{plate}-01')
        df_plates.append(df_plate)
    df_plates = pd.concat(df_plates)
    df_plates['label'] = df_plates['label'].astype(str) + '_' + df_plates['well'] + '_' + df_plates['plate']

    adata = mdata[base_modality].copy()

    for imputation in imputations:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"{mlp_dir}/{base_modality}_{imputation}_{timestamp}"
        print(f"Logging to {log_dir}")

        embeddings, marker_ids, intensities, obs_df = data_setup(adata, df_stain_layout, df_plates, imputation, stain_dict)

        N = embeddings.shape[0]
        DAPI_DIM = embeddings.shape[1]
        NUM_MARKERS = len(stains)

        train_idx, val_idx, test_idx = get_train_val_test_indices(len(embeddings))

        train_loader = dataloader_from_indices(embeddings, marker_ids, intensities, train_idx, batch_size=256, shuffle=True)
        val_loader   = dataloader_from_indices(embeddings, marker_ids, intensities, val_idx, batch_size=256, shuffle=False)

        model = ConditionalMarkerMLP(
            dapi_dim=DAPI_DIM,
            num_markers=NUM_MARKERS,
            hidden_dims=[512, 256, 128],
            dropout=0.1,
        )

        model = train(model, train_loader, val_loader, epochs=epochs, log_dir=log_dir)
        torch.save(model.state_dict(), f"{mlp_dir}/{base_modality}_{imputation}_{timestamp}.pth")
        print(f'MLP for {imputation} imputation saved to {mlp_dir}/{base_modality}_{imputation}_{timestamp}.pth')