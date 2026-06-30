# source utils
source('analysis/pilotscreen/utils.R')
# load libraries
library(ggridges)
library(RColorBrewer)
library(ggraph)
library(tidygraph)
library(ggrepel)

# set directories
dir_screen <- 'data/processed/tumoroidscreen'
dir_adata <- str_c(dir_screen, 'anndata', sep='/')
dir_plots <- str_c(dir_screen, 'plots', sep='/')
dir_analysis <- 'analysis/tumoroidscreen'
dir_metafiles <- 'metafiles'

# read mdata
mdata_org <- read_mdata(str_c(dir_adata, 'mdata_org_combined.h5mu', sep = '/'))
# read genome info
df_genes_hs <- read_tsv(str_c(dir_analysis,'tables','Homo_sapiens_gene_info.tsv', sep='/'))
# read pdac_de_genes
df_de_caf <- readxl::read_excel(str_c(dir_analysis,'tables','pdac_caf_degenes_wc2.xlsx', sep = '/')) %>%
  rename(gene=`...1`) %>% mutate(score=logfc * `-log10p`)
df_de_cancer <- readxl::read_excel(str_c(dir_analysis,'tables','pdac_cancer_degenes_wc2.xlsx', sep = '/')) %>%
   rename(gene=`...1`) %>% mutate(score=logfc * `-log10p`)
# read panther pathway attributes
panther_attr_file <- str_c(dir_analysis,'tables','gene_attribute_matrix.txt',sep='/')
df_panther <- read_tsv(panther_attr_file)
colnames(df_panther) <- c('GeneSymbol','UniprotAAC','GeneID', colnames(df_panther)[-(1:3)])
df_panther <- df_panther[-(1:2),] %>% mutate_at(colnames(df_panther)[-(1:3)], as.integer)

# read panther enrichments
read_enrichment <- function(file){
  read_tsv(file, skip = 11) %>% slice(-n()) %>%
  separate(col='PANTHER Pathways', into=c('PANTHER_Pathway','PANTHER_ID'), sep=str_escape(' (')) %>%
  mutate(PANTHER_ID=str_remove(PANTHER_ID, str_escape(')')))
}
panther_compound_enrichment_file <- str_c(dir_analysis,'tables','panther_analysis_compound_library.txt',sep='/')
panther_hits_enrichment_file <- str_c(dir_analysis,'tables','panther_analysis_hits.txt', sep='/')
df_compound_enrichment <- read_enrichment(panther_compound_enrichment_file)
df_hits_enrichment <- read_enrichment(panther_hits_enrichment_file)
colnames(df_hits_enrichment) <- str_remove(colnames(df_hits_enrichment),str_escape('upload_1 (')) %>%
  str_remove(str_escape(')')) %>% str_remove(str_escape('.tsv - REFLIST (1181'))

# load average intensity data
df_plates <- list.files(dir_screen) %>%
  map_df(~{
    if (!str_detect(.x, 'HM0')) {
      return(NULL) }
    if (file.exists(file.path(dir_screen, .x, 'plate_information.csv'))) {
      readr::read_csv(file.path(dir_screen, .x, 'plate_information.csv'), col_types = cols()) }
    else {
      return(NULL)}
  })

df_summary <- map(df_plates$dir_processed %>% set_names(df_plates$plate_id),
                  ~{readr::read_csv(file.path(.x, 'features', 'nuclei', 'df_summary_TIF_OVR_BG.csv'),
                                    col_types = cols())}) %>%
  bind_rows(.id = 'plate_id') %>%
  mutate(z_stack = as.integer(z_stack)) %>%
  mutate(plate = str_remove(plate_id, '-\\d\\d')) %>%
  rename(well_id = well)

# load compound metadata
df_plate_layouts <- tibble(plate = c('HM001', 'HM002', 'HM003', 'HM004', 'HM005', 'HM006'),
                           PickSet = c('RyoPlate1', 'RyoPlate1', 'RyoPlate2', 'RyoPlate2', 'RyoPlate3', 'RyoPlate3'))
df_plate_layout <- readr::read_csv(str_c(dir_metafiles,'plate_layout_dummy.csv', sep='/')) %>%
  rename(row = plateRow, column = plateColumn) %>%
  mutate(well = str_c(LETTERS[row], sprintf("%02d", column)))
df_drugs <- readxl::read_excel(str_c(dir_metafiles,'RO_lib_layouts_withMeta.xlsx', sep='/')) %>%
  left_join(df_plate_layouts, relationship = 'many-to-many') %>%
  mutate(well = str_c(LETTERS[plateRow_dest], sprintf("%02d", plateColumn_dest))) %>%
  rename(plate_id = plate, well_id = well)

correct_gene_symbols <- function(gene_ids){
  if (is.na(gene_ids)){
    return(NA)
  }
  if (str_detect(gene_ids,',')){
   gene_ids <- gene_ids %>% str_replace_all(',',';')
  }
  gene_ids <- gene_ids %>% str_split(pattern=';') %>% unlist() %>% as.integer()
  gene_symbols <- df_genes_hs %>% filter(GeneID %in% gene_ids) %>% pull(Symbol) %>% str_flatten(collapse = ';')
  return(gene_symbols)
}
df_drugs <- df_drugs %>% rowwise() %>% mutate(GeneSymbolCorr = correct_gene_symbols(GeneID)) %>% ungroup() %>%
  mutate(GeneSymbol = ifelse(is.na(GeneSymbolCorr)|GeneSymbolCorr=='', GeneSymbol, GeneSymbolCorr))



# merge to df_summary
df_summary <- df_summary %>%
  left_join(df_drugs, by = c('plate_id', 'well_id')) %>%
  mutate(group = ifelse(is.na(PickSet), 'control', 'drug'),
         cycle = str_sub(plate_id, -2, -1))

# extract organoid embedding from mdata
df_plot_umap <- mdata_org['phenocoder_combined']$obs %>%
as_tibble() %>%
bind_cols(mdata_org['phenocoder_combined']$obsm$X_umap %>%
        as_tibble() %>%
        rename(UMAP1 = V1, UMAP2 = V2))  %>% left_join(df_drugs) %>%
  mutate(compound = ifelse(negative_control == 'True', 'DMSO', CODENAME))

# plot plate layout with mean_intensities
prepare_df_plot <- function(id, df) {
  df_tmp <- df %>%
    filter(plate_id == id) %>%
    group_by(channel, well_id) %>%
    summarise(mean_intensity = mean(mean, na.rm = TRUE))
  channels <- df_tmp %>%
    distinct(channel) %>%
    pull() %>%
    set_names()
  df_plot <- map(channels, function(x) { df_plate_layout %>%
    rename(well_id = well) %>%
    mutate(channel = x) }) %>%
    bind_rows() %>%
    left_join(df_tmp, by = c('well_id', 'channel')) %>%
    group_by(channel) %>%
    mutate(mean_intensity = rescale(mean_intensity)) %>%
    ungroup() %>%
    mutate(channel = str_c('ch_', channel))
  return(df_plot)
}

plates <- df_summary %>%
  distinct(plate_id) %>%
  pull() %>%
  set_names()

df_plot <- map(plates, prepare_df_plot, df = df_summary) %>%
  bind_rows(.id = 'plate_id') %>%
  separate(plate_id, into = c('plate', 'cycle'), remove = FALSE) %>%
  filter(plate %in% c(df_plot_umap %>% distinct(plate_id) %>% pull()))


df_cluster_intensities <- df_plot %>%
  pivot_wider(id_cols = c('plate', 'well_id'), values_from = mean_intensity, names_from = c('cycle', 'channel')) %>%
  na.omit()

df_cluster_intensities$cluster <- kmeans(df_cluster_intensities %>%
                                           select(-plate, -well_id) %>%
                                           as.matrix(), centers = 3)$cluster

df_plot <- df_plot %>% left_join(df_cluster_intensities %>% select(plate, well_id, cluster))


p_mean_intensities <- ggplot(df_plot %>% filter(cycle %in% c('01', '03'), !(plate == 'HM003' & column < 19 & row %% 2 == 0)),
                             aes(x = column, y = row, fill = mean_intensity)) +
  geom_raster() +
  facet_grid(plate_id ~ channel) +
  coord_equal() +
  scale_y_reverse() +
  scale_fill_viridis_c(option = 'A') +
  theme_minimal()

p_mean_intensities_cluster <- ggplot(df_plot, aes(x = column, y = row, fill = as.factor(cluster))) +
  geom_raster() +
  facet_grid(plate_id ~ channel) +
  coord_equal() +
  scale_y_reverse() +
  theme_minimal()

p_cluster_plates <- ggplot(df_plot_umap, aes(x = as.integer(column), y = as.integer(row), fill = leiden)) +
  geom_raster() +
  facet_wrap(~plate_id) +
  coord_equal() +
  scale_y_reverse() +
  scale_fill_brewer(palette = 'Set3') +
  theme_minimal()

p_cluster_plates_filtered <- ggplot(df_plot_umap %>%
                                      filter(!(plate_id == 'HM003' & as.integer(column) < 19 & as.integer(row) %% 2 == 0),
                                             leiden!='8'),
                                             aes(x = as.integer(column), y = as.integer(row), fill = leiden)) +
                                      geom_raster() +
                                      facet_wrap(~plate_id) +
                                      coord_equal() +
                                      scale_y_reverse() +
                                      scale_fill_brewer(palette = 'Set3') +
                                      theme_minimal()

# mahalanobis distances
prepare_data_mahal <- function(mdata_org) {
df_features <- mdata_org['phenocoder_combined']$obs %>%
as_tibble() %>%
bind_cols(as_tibble(mdata_org['phenocoder_combined']$obsm$X_pca) %>%
          rename(all_of(colnames(.) %>% set_names(str_c('pca_', 1:32))))) %>%
bind_cols(mdata_org['phenocoder_combined']$obsm$X_umap %>%
          as_tibble() %>%
          rename(UMAP1 = V1, UMAP2 = V2)) %>%
left_join(df_drugs) %>%
mutate(compound = ifelse(negative_control == 'True', 'DMSO', CODENAME))
return(list(df_features = df_features))
}

list_mahal <- prepare_data_mahal(mdata_org)
list_mahal$df_features <- list_mahal$df_features %>% filter(!(plate_id == 'HM003' &
  as.integer(column) < 19 &
  as.integer(row) %% 2 == 0),
                                                            leiden != '8')
list_mahal$mahal_results <- pairwise.mahalanobis(list_mahal$df_features %>%
                                                   select(starts_with('pca')) %>%
                                                   as.matrix(), list_mahal$df_features$compound)

list_mahal$mahal_results$distance['DMSO',] %>%
  enframe(name = 'compound', value = 'distance_dmso') %>%
  #filter(compound != 'DMSO') %>%
  mutate(compound_factor = factor(compound, levels = compound[order(distance_dmso)]),
         compound_int = as.integer(compound_factor)) -> df_dmso_mahal

p_rank_compounds <- ggplot(df_dmso_mahal %>% filter(compound != 'DMSO'), aes(x = compound_int, y = log1p(distance_dmso))) +
  geom_point(size = 0.5) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title = element_blank()) +
  scale_y_continuous(breaks = c(4, 5, 6, 7, 8, 9))

ggsave(plot = p_rank_compounds, filename = str_c(dir_plots, 'compound_ranks.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)

p_hist <- ggplot(df_dmso_mahal %>% filter(compound != 'DMSO'), aes(x = log1p(distance_dmso))) +
  geom_histogram(col = 'black', fill = 'grey', bins = 30, size = 0.2) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title = element_blank()) +
  scale_x_continuous(expand = c(0, 0), breaks = c(4, 5, 6, 7, 8, 9)) +
  scale_y_continuous(expand = c(0, 0))

ggsave(plot = p_hist, filename = str_c(dir_plots, 'compound_histogram_distance.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)

p_umap_ctrl <- ggplot(NULL, aes(UMAP1, UMAP2)) +
  geom_point(data = list_mahal$df_features %>% filter(negative_control == 'False'),
             fill = alpha("black", 0.1), col = 'grey90', size = 0.5) +
  geom_point(data = list_mahal$df_features %>% filter(negative_control == 'True'),
             fill = "#d62728", size = 1.5, shape = 21, col = 'black', stroke = 0.25) +
  theme_void()

ggsave(plot = p_umap_ctrl, filename = str_c(dir_plots, 'umap_ctrl.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)

p_umap_cluster <- ggplot(list_mahal$df_features, aes(UMAP1, UMAP2, fill = leiden)) +
  geom_point(size = 1.5, shape = 21, col = 'black', stroke = 0.1) +
  theme_void() +
  scale_fill_brewer(palette = 'Set3')

ggsave(plot = p_umap_cluster, filename = str_c(dir_plots, 'umap_cluster_with_legend.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)

ggsave(plot = p_umap_cluster + theme(legend.position = 'None'),
       filename = str_c(dir_plots, 'umap_cluster_without_legend.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)


p_umap_distance <- ggplot(NULL, aes(UMAP1, UMAP2)) +
  geom_point(data = list_mahal$df_features %>% filter(negative_control == 'True'),
             fill = alpha("black", 0.1), col = 'grey70', size = 1.5) +
  geom_point(data = list_mahal$df_features %>%
    left_join(df_dmso_mahal) %>%
    filter(negative_control == 'False'),
             aes(fill = log1p(distance_dmso)), size = 1.5, col = 'black', stroke = 0.1, shape = 21) +
  theme_void() +
  scale_fill_viridis_c(option = 'magma')

ggsave(plot = p_umap_distance, filename = str_c(dir_plots, 'umap_distance_with_legend.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)

ggsave(plot = p_umap_distance + theme(legend.position = 'None'),
       filename = str_c(dir_plots, 'umap_distance_without_legend.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)


p5 <- ggplot(list_mahal$df_features, aes(UMAP1, UMAP2, col = plate_id)) +
  geom_point() +
  theme_bw()

df_compounds <- list_mahal$df_features %>%
left_join(df_dmso_mahal)

compounds <- df_compounds %>%
  filter(compound!='DMSO') %>%
  select(compound) %>%
  distinct() %>%
  pull()
df_genes <- df_compounds %>%
  select(GeneSymbol) %>%
  na.omit() %>% pull() %>%
  str_split(pattern = ';') %>%
  enframe(value = 'GeneSymbol') %>%
  unnest(cols=colnames(.)) %>%
  select(-name)
genes <- df_genes %>% distinct() %>% filter(GeneSymbol!='') %>% arrange(GeneSymbol) %>% pull()
df_genes <- df_genes %>% group_by(GeneSymbol) %>% summarise(n=n())
# write df_genes GeneSymbol to .tsv
df_genes %>% select(GeneSymbol) %>% write_tsv(str_c(dir_analysis,'tables','compound_library_genes.tsv',sep='/'))





# Initialize a matrix with NA values
gene_matrix <- matrix(0, nrow = length(compounds), ncol = length(genes),
                      dimnames = list(compounds, genes))

# Populate the matrix with distance to DMSO value where a gene is associated with a compound
for (i in seq_along(compounds)) {
  compound_genes <- df_compounds %>%
    filter(compound == compounds[i]) %>%
    select(GeneSymbol) %>%
    na.omit() %>%
    pull() %>%
    str_split(pattern = ';') %>%
    unlist()
  distance_dmso <- df_compounds %>%
    filter(compound == compounds[i]) %>%
    select(distance_dmso) %>%
    pull() %>%
    unique()
  gene_matrix[i, compound_genes] <- distance_dmso
}

# Convert the matrix to a data frame for better readability
df_gene_matrix <- as.tibble(gene_matrix, rownames = 'compound')
df_gene_ranks <- df_gene_matrix %>% select(-compound) %>% summarise_all(sum) %>% pivot_longer(cols = colnames(.)) %>%
  rename(gene=name, distance_sum=value) %>%
  left_join(df_genes %>% rename(gene=GeneSymbol)) %>%
  mutate(distance_sum_log = log1p(distance_sum),
         gene_rank_sum=rank(distance_sum),
         distance_sum_norm = distance_sum/n,
         gene_rank_norm = rank(distance_sum_norm, ties.method = 'random'))
top_genes <- df_gene_ranks %>%
  arrange(desc(distance_sum_norm)) %>%
  head(100)
# write df_gene_ranks
df_gene_ranks %>%
  arrange(desc(distance_sum_norm)) %>%
  mutate(top_100 = ifelse(gene_rank_norm - max(gene_rank_norm) <=100,1,0)) %>%
  write.csv(str_c(dir_analysis,'tables','top100_genes_sorted.csv',sep='/'))



p_gene_ranks <- ggplot(df_gene_ranks %>% filter(gene_rank_norm > nrow(df_gene_ranks) - 100),
                       aes(x=gene_rank_norm,y=distance_sum_norm)) +
  geom_point() +
  coord_cartesian(clip = "off") +
  theme_bw() +
  geom_label_repel(data = top_genes, aes(label = gene), min.segment.length = 0, max.overlaps = 100,
                   xlim = c(-Inf, Inf), ylim = c(-Inf, Inf),fill = "white") +
  guides(fill = guide_legend(override.aes = aes(label = ""))) +
  theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(), axis.title.x = element_blank())
p_gene_ranks_all <- ggplot(df_gene_ranks , aes(x=gene_rank_norm,y=log1p(distance_sum_norm))) +

  geom_vline(xintercept = nrow(df_gene_ranks) - 100, color='grey', linetype="dotted") +
  geom_point(size=0.5) +
  geom_label_repel(data = top_genes %>% slice(1:5), aes(label = gene), min.segment.length = 0, max.overlaps = 100,
                   xlim = c(-Inf, Inf), ylim = c(-Inf, Inf),fill = "white", size=2) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title = element_blank()) +
  scale_y_continuous(breaks = c(4, 5, 6, 7, 8, 9))
ggsave(plot = p_gene_ranks_all, filename = str_c(dir_plots, 'gene_ranks.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 50, height = 50)

# select compounds for creating example overlays
df_drugs %>% filter(CODENAME %in% (gene_matrix[rowSums(gene_matrix[,top_genes %>% slice(1:20) %>% pull(gene)]) > 0,top_genes %>% slice(1:20) %>% pull(gene)] %>% rownames())) %>% distinct() -> df_hits_selected
df_hits_selected %>% left_join(df_compounds) %>% arrange(distance_dmso) %>% select(colnames(df_drugs), distance_dmso) %>% View()


# get gene set unions and scores
genes_common_caf <- genes[genes %in% (df_de_caf %>% filter(score>0) %>% pull(gene))]
genes_common_cancer <- genes[genes %in% (df_de_cancer %>% filter(score>0) %>% pull(gene))]
scores_cancer <- df_de_cancer[df_de_cancer$gene %in% genes_common_cancer,] %>% pull(score, gene)
scores_caf <- df_de_caf[df_de_caf$gene %in% genes_common_caf,] %>% pull(score, gene)

score_perturbations <- function(M, df_de, threshold_score, normalize_adj=TRUE){
  genes <- colnames(M)
  genes_common <- genes[genes %in% (df_de %>% filter(score>=threshold_score) %>% pull(gene))]
  scores <- df_de[df_de$gene %in% genes_common,] %>% pull(score, gene) %>% rescale()
  M <- M[,genes_common]
  M <- log1p(M)
  A <- M[,genes_common]
  A[A>0] <- 1
  if (normalize_adj) {
    A <- A/rowSums(A)
    A[is.na(A)] <- 0
  }
  scores_perturb <- (M*A) %*% scores
  scores_norm <-  A %*% scores
  df <- as.tibble(scores_perturb, rownames = 'compound') %>% rename(score_perturb=V1) %>%
    left_join(as.tibble(scores_norm, rownames='compound') %>% rename(score_norm=V1))
  return(list('df'=df,'M'=M,'A'=A))
}

perturb_caf <- score_perturbations(gene_matrix,df_de_caf,1)
perturb_cancer <- score_perturbations(gene_matrix,df_de_cancer,1)

df_perturb_ct <- left_join(perturb_caf$df,perturb_cancer$df, by='compound', suffix = c('_caf','_cancer')) %>%
  left_join(df_dmso_mahal %>% select(compound,distance_dmso)) %>%
  mutate(score_perturb = score_perturb_cancer - score_perturb_caf,
         score_norm = score_norm_cancer - score_norm_caf,
         score_diff = abs(score_perturb - score_norm),
         score_dist = score_perturb_cancer*score_perturb_caf,
         score_all = sqrt(score_dist^2+log1p(distance_dmso)^2)) %>%
  left_join(df_gene_matrix %>% select(compound, MAP2K1, LPAR1))

p_diff <- ggplot(df_perturb_ct, aes(x=score_perturb,y=log1p(distance_dmso),
                          fill=log1p(distance_dmso), size=log1p(score_diff))) +
  geom_point(col='black',shape=21) + theme_bw() +
  scale_fill_viridis_c(option='magma')

p_perturb <- ggplot(df_perturb_ct, aes(x=score_perturb_cancer,y=score_perturb_caf,
                               fill=log1p(distance_dmso), size=log1p(distance_dmso))) +
  geom_point(col='black',shape=21) + theme_bw() +
  scale_fill_viridis_c(option='magma')
p <- ggplot(df_perturb_ct, aes(y=score_dist, x=log1p(distance_dmso))) +
  geom_point() + theme_bw()
p_hist_score_aal <- ggplot(df_perturb_ct, aes(x=score_all)) + geom_histogram() + theme_bw()
p + p_diff + p_perturb + p_hist



# plot genes scores
df_plot_gene <- df_de_caf %>%
  left_join(df_de_cancer, by='gene', suffix = c('_caf','_cancer')) %>%
  left_join(df_gene_ranks %>% select(gene,distance_sum_norm), by='gene') %>%
  mutate(score_cancer_log = ifelse(score_cancer != 0, ifelse(score_cancer < 0, -log10(abs(score_cancer)), log10(score_cancer)),0),
         score_caf_log=ifelse(score_caf != 0, ifelse(score_caf < 0, -log10(abs(score_caf)), log10(score_caf)),0))

df_plot_subset <- df_plot_gene %>% filter(!is.na(distance_sum_norm))
ggplot(data=NULL) +
    geom_point(data=df_plot_gene,aes(x=score_cancer_log, y=score_caf_log), col='grey70', alpha=0.2) +
    geom_point(data = df_plot_subset %>% arrange(distance_sum_norm),
               aes(x=score_cancer_log, y=score_caf_log,col=log1p(distance_sum_norm), size=distance_sum_norm)) +
  coord_equal() + theme_bw() + scale_color_viridis_c(option='magma') +
  xlim(c(-6,6))+
  xlim(c(-6,6))

p_scores_atlas <- ggplot(data=NULL) +
    geom_point(data=df_plot_gene,aes(x=score_cancer, y=score_caf), col='grey70', alpha=0.2) +
    geom_point(data = df_plot_subset %>% arrange(distance_sum_norm),
               aes(x=score_cancer, y=score_caf,col=log1p(distance_sum_norm), size=distance_sum_norm)) +
  coord_equal() + theme_bw() + scale_color_viridis_c(option='magma') +
  xlim(c(-60,60))+
  ylim(c(-60,60))+
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_blank())
ggsave(plot = p_scores_atlas + theme(legend.position = 'none'),
       filename = str_c(dir_plots, 'scores_atlas.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 200, height = 200)

ggsave(plot = p_scores_atlas + theme(legend.position = 'none'),
       filename = str_c(dir_plots, 'scores_atlas.png', sep = '/'),
       dpi = 300, units = 'mm', width = 200, height = 200)



M_plot <- log1p(gene_matrix)
M_plot <- M_plot[df_dmso_mahal %>% filter(compound_int > max(compound_int) - 50) %>% pull(compound),top_genes$gene]
heatmap <- list_mahal$mahal_results$distance %>% log1p() %>% pheatmap::pheatmap(silent = TRUE, show_colnames = FALSE, show_rownames = FALSE, scale = 'none')
heatmap <- M_plot %>% pheatmap::pheatmap(silent = TRUE, show_colnames = FALSE, show_rownames = FALSE, scale = 'none')
ggsave(plot=grid.arrange(heatmap[[4]]),filename = str_c(dir_plots,'heatmap.png', sep = '/'),limitsize = FALSE)



pca_results <- prcomp(gene_matrix, rank. = 10)
library(umap)
umap_embedding <- umap(as.data.frame(pca_results$rotation))

df_gene_embedding <- as.tibble(pca_results$rotation, rownames='gene') %>%
  left_join(as.tibble(umap_embedding$layout, rownames='gene') %>% rename(UMAP1=V1, UMAP2=V2)) %>%
  left_join(colSums(gene_matrix) %>%
              enframe(name='gene', value='dmso_effect') %>%
              left_join(df_genes %>% rename(gene=GeneSymbol)) %>%
              mutate(dmso_effect_norm = dmso_effect / n))
df_gene_edges <- gene_matrix %>% t() %>% dist() %>% as.matrix() %>% as.tibble(rownames = 'from') %>%
  pivot_longer(names_to = 'to', cols=.$from, values_to = 'weight') %>%
  filter(to!=from) %>%
  mutate(weight=max(weight)-weight) %>%
  filter(weight >  19453.85)


g_genes <- tbl_graph(nodes = df_gene_embedding, df_gene_edges)
p_graph_genes <- ggraph(g_genes, x=UMAP, y=UMAP) +
  geom_edge_fan(aes(alpha=weight), show.legend = FALSE, edge_color='gray') +
  geom_node_point(aes(size=dmso_effect_norm, fill=log1p(dmso_effect_norm)),shape = 21, color = 'black') +
  scale_size_continuous(range = c(1, 20)) +
  scale_alpha_continuous(range=c(0,0.05)) +
  scale_fill_viridis_c(option='magma') +
  theme_void() +
  theme(legend.position = "right")

ggsave(dpi=72, plot = p_graph_genes, filename = str_c(dir_plots,'gene_graph.png', sep = '/'))

# aggregate genes x compound matrix into pathways x compound matrix
panther_matrix <- df_panther[,-c(2,3)][,-1] %>% as.matrix()
rownames(panther_matrix) <- df_panther$GeneSymbol
# select compound library enriched pathways
panther_matrix <- panther_matrix[,panther_matrix %>% colnames() %in% df_compound_enrichment$PANTHER_Pathway]
panther_matrix <- t(panther_matrix)
# subset panther matrix for genes in gene_matrix
common_genes <- intersect(colnames(panther_matrix), colnames(gene_matrix))
panther_matrix <- panther_matrix[, common_genes]
gene_matrix_subset <- gene_matrix[,common_genes]

# dot product gene_matrix x panther_matrix
pathway_dot_product <- gene_matrix_subset %*% t(panther_matrix)
df_pathway_perturb <- colSums(pathway_dot_product) %>% enframe(name='pathway',value='distance_sum') %>% mutate(distance_log1p = log1p(distance_sum))
df_hits_enrichment %>% rename(pathway=PANTHER_Pathway) %>% left_join(df_pathway_perturb) -> df_hits_enrichment
 df_hits_enrichment$pathway <- factor(df_hits_enrichment$pathway, levels = df_hits_enrichment$pathway[order(df_hits_enrichment$`fold Enrichment`)], ordered=TRUE)

p_enrichment <- ggplot(df_hits_enrichment, aes(y=pathway, x=`fold Enrichment`, fill=distance_sum)) + geom_bar(stat='identity') + scale_fill_viridis_c(option = 'magma') + theme_bw() + theme(axis.title.y = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position = 'none') + scale_x_continuous(expand=c(0,0))


ggsave(plot = p_enrichment,
       filename = str_c(dir_plots, 'enrichment.pdf', sep = '/'),
       dpi = 72, units = 'mm', width =200, height = 90)

# pca
pathway_dot_product <- gene_matrix_subset %*% t(panther_matrix)
pca_results <- prcomp(pathway_dot_product, rank. = 10)
library(umap)
umap_embedding <- umap(as.data.frame(pca_results$rotation))

df_graph_embedding <- as.tibble(pca_results$rotation, rownames='pathway') %>%
  left_join(as.tibble(umap_embedding$layout, rownames='pathway') %>% rename(UMAP1=V1, UMAP2=V2)) %>%
  left_join(as.tibble(enframe(colSums(pathway_dot_product), name='pathway', value='distance_sum'))) %>%
  mutate(distance_sum_log1p = log1p(distance_sum))

df_graph_edges <- df_panther %>%
  filter(GeneSymbol %in% common_genes) %>%
  select(c('GeneSymbol',df_graph_embedding$pathway))%>%
  select(-GeneSymbol)%>%
  as.matrix() %>%
  t() %>% dist() %>% as.matrix() %>% as.tibble(rownames = 'from') %>%
  pivot_longer(names_to = 'to', cols=.$from, values_to = 'weight') %>%
  filter(to!=from) %>%
  mutate(weight=max(weight)-weight)
g <- tbl_graph(edges=df_graph_edges,nodes=df_graph_embedding, directed = FALSE)

p <- ggplot(df_graph_embedding, aes(x=UMAP1, y=UMAP2, fill=distance_sum_log1p)) +
  geom_point(aes(size=distance_sum), shape=21, col='black') +
  coord_cartesian(clip = "off") +
  theme_bw() +
  theme(panel.grid = element_blank()) +
  scale_fill_viridis_c() +
  geom_label_repel(data=df_hits_enrichment %>% rename(pathway=PANTHER_Pathway) %>% left_join(df_graph_embedding),
                   aes(label=pathway), min.segment.length = 0 , max.overlaps = 10,
                   xlim = c(-Inf, Inf), ylim = c(-Inf, Inf),fill = "white") +
  guides(fill = guide_legend(override.aes = aes(label = ""))) +
  scale_size_continuous(range=c(1, 20))


ggsave(dpi=72, plot = p, filename = str_c(dir_plots,'graph.png', sep = '/'))

library(tidygraph)
library(ggraph)
df_graph_edges <- df_panther %>%
  filter(GeneSymbol %in% common_genes) %>%
  select(c('GeneSymbol',df_graph_embedding$pathway))%>%
  select(-GeneSymbol)%>%
  as.matrix() %>%
  t() %>% dist() %>% as.matrix() %>% as.tibble(rownames = 'from') %>%
  pivot_longer(names_to = 'to', cols=.$from, values_to = 'weight') %>%
  filter(to!=from) %>%
  mutate(weight=max(weight)-weight) %>%
  filter(weight>6)
g <- tbl_graph(edges=df_graph_edges,nodes=df_graph_embedding, directed = FALSE)

p_tidygraph <- ggraph(g, x=UMAP1, y=UMAP2) +
  geom_edge_fan(aes(alpha=weight), show.legend = FALSE, edge_color='gray') +
  geom_node_point(aes(fill = distance_sum_log1p, size = distance_sum), shape = 21, color = 'black') +
  geom_node_label(aes(filter=distance_sum_log1p > 10.5,label = pathway), repel = TRUE, size = 5) +
  scale_fill_viridis_c(option='magma') +
  scale_size_continuous(range = c(1, 20)) +
  scale_alpha_continuous(range=c(0,0.05)) +
  theme_void() +
  theme(legend.position = "right")

p_tidygraph
ggsave(dpi=72, plot = p_tidygraph, filename = str_c(dir_plots,'graph.pdf', sep = '/'))
