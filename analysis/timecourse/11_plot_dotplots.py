import matplotlib.pyplot as plt
import muon as mu
import scanpy as sc
from scipy.cluster import hierarchy as sch

plt.rc("pdf", fonttype=42)
sc._settings.settings._vector_friendly = True


def order_genes(adata, preselected_genes=None):
    if preselected_genes is not None:
        adata = adata[:, preselected_genes]
    Z = sch.linkage(adata.X.T, method="ward")
    dendrogram = sch.dendrogram(Z, no_plot=True)
    ordered_genes = [adata.var_names[i] for i in dendrogram["leaves"]]
    return ordered_genes


def plot_dotplots(adata, adata_neighbors, output_dir, ordered_genes=None):
    if ordered_genes is None:
        ordered_genes = order_genes(adata)

    # Create dotplots for both datasets using the same gene order
    for data, data_name in [(adata, "nuclei"), (adata_neighbors, "neighbors")]:
        dp = sc.pl.dotplot(
            data,
            var_names=ordered_genes,
            groupby="leiden",
            dendrogram=True,  # Enable hierarchical clustering of rows
            return_fig=True,
        )
        dp.add_totals(color="grey").style(
            dot_edge_color="black",
            dot_edge_lw=0.5,
            cmap="Greys",
            dot_max=0.5,
            dot_min=0.1,
        )
        dp.savefig(f"{output_dir}/dotplot_{data_name}.pdf")
        plt.close("all")


mdata = mu.read_h5mu(
    "/pmount/projects/site/pred/ihb-tumoroid/data/processed/timecourse/anndata/mdata_registered_imputed_mlp_normalized.h5mu"
)
adata = mdata["phenocoder_msg_nuclei_imputed"]
adata_neigh = mdata["phenocoder_msg_neighbors_imputed"]
order_genes = order_genes(adata)
sc.set_figure_params(figsize=(20, 20))
sc.settings.figdir = "analysis/timecourse/plots"
plot_dotplots(adata, adata_neigh, sc.settings.figdir, ordered_genes=order_genes)
sc.set_figure_params(
    dpi_save=72,
    vector_friendly=True,
)
# plot umaps for SDC1, YAP and Phalloidin
sc.pl.umap(
    adata,
    color=["SDC1", "YAP", "Phalloidin"],
    size=0.2,
    vmin="p1",
    vmax="p99",
    save="_SDC1_YAP_Phalloidin.pdf",
)

sc.pl.umap(
    adata_neigh,
    color=["SDC1", "YAP", "Phalloidin"],
    size=0.2,
    vmin="p1",
    vmax="p99",
    save="_SDC1_YAP_Phalloidin.pdf",
)
