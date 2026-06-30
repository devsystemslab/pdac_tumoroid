from pathlib import Path

import muon as mu
import pandas as pd
import plotly.express as px
import scanpy as sc
import yaml

screen = "timecourse"
file = "configs/params.yaml"

with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

df_plate_layout = pd.read_csv(Path(params["dir_screen"], "timecourse_layout.csv"))
df_plate_layout = df_plate_layout.melt(
    id_vars=["row"], var_name="col", value_name="staining_set"
)
df_plate_layout["column"] = df_plate_layout["col"].str.zfill(2)
df_plate_layout["row_num"] = df_plate_layout["row"].apply(
    lambda x: ord(x) - ord("A") + 1
)
df_plate_layout["well_id"] = df_plate_layout["row"] + df_plate_layout["column"]
df_plate_layout = df_plate_layout.sort_values(by=["well_id"])

dir_screen = params["dir_screen"]
plates = params["plates"]
df_plates = pd.concat(
    [
        pd.read_csv(
            f"{dir_screen}/{plate}/plate_information.csv", dtype={plate: str}
        ).assign(plate=plate)
        for plate in plates
    ]
)

df_plate_layouts = df_plate_layout.merge(df_plates, how="cross")
df_plate_layouts["plate_id"] = df_plate_layouts["plate"]
df_plate_layouts = df_plate_layouts.map(str)

dir_results = Path(params["dir_screen"], "anndata")

mdata_org = mu.read_h5mu(Path(dir_results, "mdata_org_newclusters.h5mu"))

obs_filter = [
    "B06_001",
    "C06_002",
    "N02_001",
    "M06_001",
    "F02_001",
    "C06_001",
    "N04_001",
    "P04_001",
    "O01_001",
]
mdata_org.mod["all_combined"] = mdata_org["all_combined"][
    ~mdata_org["all_combined"].obs.index.isin(obs_filter), :
]

sc.pl.pca(mdata_org["all_combined"], color=["plate_id"])

df = pd.DataFrame(
    {
        "UMAP1": mdata_org["all_combined"].obsm["X_pca"][:, 0],
        "UMAP2": mdata_org["all_combined"].obsm["X_pca"][:, 1],
        "well_id": mdata_org["all_combined"].obs["well_id"],
        "plate_id": mdata_org["all_combined"].obs["plate_id"],
    }
)

# Create interactive plot
fig = px.scatter(
    df,
    x="UMAP1",
    y="UMAP2",
    color="plate_id",
    hover_data=["well_id"],
    title="UMAP with Sample Labels",
)
fig.show()

sc.tl.diffmap(mdata_org["all_combined"])
print(mdata_org["phenocoder"].obsm["X_diffmap"].shape)

sc.pl.scatter(mdata_org["all_combined"], basis="diffmap", color=["plate_id"])

# select as root cell the cell with the most extreme value in diffmap dimension
root_ixs = mdata_org["all_combined"].obsm["X_diffmap"][:, 1].argmin()
mdata_org["all_combined"].uns["iroot"] = root_ixs

sc.tl.dpt(mdata_org["all_combined"])
sc.pl.umap(mdata_org["all_combined"], color=["dpt_pseudotime", "plate_id"])

mdata_org.write_h5mu(Path(dir_results, "mdata_org_newclusters.h5mu"))
