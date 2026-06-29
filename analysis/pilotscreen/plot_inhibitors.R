# source utils
source('whole_mount_tumoroid/analysis/inhibitors/utils.R')

# load libraries
library(ggridges)
library(RColorBrewer)

# colors
colors_compound <- c(`Ac-Gly-BoroPro` = "#1f77b4", Bortezomib = "#ff7f0e",
                     `BTT-3033` = "#279e68", DMSO = "#d62728", Erlotinib = "#aa40fc",
                     Gemcitabine = "#8c564b", Ilomastat = "#e377c2", `LGK-974` = "#b5bd61",
                     Linsitinib = "#17becf", Paclitaxel = "#aec7e8", `PF-562271` = "#ffbb78",
                     SN38 = "#98df8a", T0070907 = "#ff9896", Trametinib = "#c5b0d5",
                     VER155008 = "#c49c94")

# set directories
dir_screen <- 'data/processed/inhibitors'
dir_adata <- 'data/processed/inhibitors/anndata'
dir_plots <- 'data/processed/inhibitors/plots'

# read mdata
mdata_org <- read_mdata(str_c(dir_adata,'mdata_org_combined.h5mu', sep='/'))
mdata_reg <- read_mdata(str_c(dir_adata,'mdata_registered.h5mu', sep='/'))
mdata_cycle_1 <- read_mdata(str_c(dir_adata,'mdata_cycle-01.h5mu', sep='/'))
mdata_cycle_3 <- read_mdata(str_c(dir_adata,'mdata_cycle-03.h5mu', sep='/'))

# plot umaps for registered sc-dataset
list_reg <- prepare_data_for_plotting(mdata_reg)
list_reg_msg <- prepare_data_for_plotting(mdata_reg, source_mod = 'nuclei_msg', target_mod = 'phenocoder_msg')
#
p_reg_pheno <- plot_nuclei_data(list_reg, list_reg_msg, type='phenocoder', frac=0.05)
ggsave(str_c(dir_plots,'phenocoder_sc_umaps_heatmaps_registered.png', sep='/'),
       plot = p_reg_pheno,
       dpi = 112,
       unit = 'mm',
       width = 400.05,
       height = 215.22)

p_reg_nuclei <- plot_nuclei_data(list_reg, list_reg_msg, type='nuclei', frac=0.05)
ggsave(str_c(dir_plots,'nuclei_sc_umaps_heatmaps_registered.png', sep='/'),
       plot = p_reg_nuclei,
       dpi = 112,
       unit = 'mm',
       width = 400.05,
       height = 215.22)

# plot umaps for cycle 1 sc-dataset
list_cycle_1 <- prepare_data_for_plotting(mdata_cycle_1)
list_cycle_1_msg <- prepare_data_for_plotting(mdata_cycle_1, source_mod = 'nuclei_msg', target_mod = 'phenocoder_msg')

p_cycle_01_pheno <- plot_nuclei_data(list_cycle_1, list_cycle_1_msg, type='phenocoder', frac=0.05)
ggsave(str_c(dir_plots,'phenocoder_sc_umaps_heatmaps_cycle_01.png', sep='/'),
       plot = p_cycle_01_pheno,
       dpi = 112,
       unit = 'mm',
       width = 400.05,
       height = 215.22)

p_cycle_01_nuclei <- plot_nuclei_data(list_cycle_1, list_cycle_1_msg, type='nuclei', frac=0.05)
ggsave(str_c(dir_plots,'nuclei_sc_umaps_heatmaps_cycle_01.png', sep='/'),
       plot =p_cycle_01_nuclei,
       dpi = 112,
       unit = 'mm',
       width = 400.05,
       height = 215.22)

# plot umaps for cycle 3 sc-dataset
list_cycle_3 <- prepare_data_for_plotting(mdata_cycle_3)
list_cycle_3_msg <- prepare_data_for_plotting(mdata_cycle_3, source_mod = 'nuclei_msg', target_mod = 'phenocoder_msg')
p_cycle_03_pheno <- plot_nuclei_data(list_cycle_3, list_cycle_3_msg, type='phenocoder', frac=0.05)
ggsave(str_c(dir_plots,'phenocoder_sc_umaps_heatmaps_cycle_03.png', sep='/'),
       plot = p_cycle_03_pheno,
       dpi = 112,
       unit = 'mm',
       width = 400.05,
       height = 215.22)

p_cycle_03_nuclei <- plot_nuclei_data(list_cycle_3, list_cycle_3_msg, type='nuclei', frac=0.05)
ggsave(str_c(dir_plots,'nuclei_sc_umaps_heatmaps_cycle_03.png', sep='/'),
       plot =p_cycle_03_nuclei,
       dpi = 112,
       unit = 'mm',
       width = 400.05,
       height = 215.22)

# plot organoid embedding
df_plot_umap <- prepare_umap_org_data(mdata_org)
# sample example images
df_plot_umap %>%
  filter(compound != 'DMSO') %>%
  group_by(conc, timepoint, compound) %>%
  sample_n(1) %>% bind_rows(df_plot_umap %>%
  filter(compound == 'DMSO') %>% group_by(timepoint) %>%
  sample_n(14)) -> df_pos_ctrl_examples


p1 <- ggplot(df_plot_umap, aes(UMAP1, UMAP2, fill = leiden)) +
  geom_point(size = 3, shape=21, color='black') +
  coord_equal() +
  scale_fill_brewer(palette='Set3') +
  theme_void() +
  guides(color = guide_legend(ncol = 2, override.aes = list(size = 3)))

p2 <- ggplot(df_plot_umap, aes(UMAP1, UMAP2, fill = compound)) +
  geom_point(size = 3, shape=21, color='black') +
  coord_equal() +
  scale_fill_manual(values = colors_compound) +
  theme_void() +
  guides(color = guide_legend(ncol = 2, override.aes = list(size = 3)))

p3 <- ggplot(df_plot_umap, aes(UMAP1, UMAP2, fill = timepoint)) +
  geom_point(size = 3, shape=21, color='black') +
  coord_equal() +
  scale_fill_manual(values=c("#ecf8fb", "#b2e1e2", "#2ba25f") %>% set_names(levels(df_plot_umap$timepoint))) +
  guides(fill = guide_legend(override.aes = list(size = 3))) +
  theme_void()

p4 <- ggplot(df_plot_umap, aes(UMAP1, UMAP2, fill = conc)) +
  geom_point(size = 3, shape=21, color='black') +
  coord_equal() +
  scale_fill_manual(values=c("#ffffff","#ecf8fb", "#b2cde3", "#8856a7") %>% rev() %>% set_names(levels(df_plot_umap$conc))) +
  guides(fill = guide_legend(override.aes = list(size = 3))) +
  theme_void()

p5 <- ggplot(df_plot_umap, aes(UMAP1, UMAP2, fill = plate_id)) +
  geom_point(size = 3, shape=21, color='black') +
  coord_equal() +
  scale_fill_manual(values=c('#E59836','#709FD5') %>% set_names(levels(df_plot_umap$plate_id))) +
  guides(fill = guide_legend(override.aes = list(size = 3))) +
  theme_void()

p6 <- ggplot(data=NULL) +
  geom_point(data=df_plot_umap %>%
    filter(!(compound %in% c('DMSO'))), aes(UMAP1, UMAP2), fill='grey90',size = 3, shape=21, color='black') +
  geom_point(data=df_plot_umap %>% filter(compound %in% c('DMSO')) %>% arrange(desc(compound)),
                                          aes(UMAP1, UMAP2, fill = compound), size = 3, shape=21, color='black') +
  coord_equal() +
  scale_fill_manual(values = colors_compound) +
  guides(fill = guide_legend(override.aes = list(size = 3))) +
  theme_void()

p_umaps <- (p1 / p2/ p6) | (p3 / p4 / p5)
ggsave('data/processed/inhibitors/plots/organoid_embeddings_umaps_only.pdf', plot = p_umaps,
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)
ggsave('data/processed/inhibitors/plots/organoid_embeddings_umaps_only.png', plot = p_umaps,
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)
# plot organoid embedding - data handling, distances
cluster_data <- prepare_cluster_data(mdata_org)
order_leiden <- get_cluster_order(cluster_data)
order_compounds <- get_compound_order(cluster_data)
order_compounds_figure <- c('Bortezomib','Trametinib','SN38',
                            'BTT-3033','Gemcitabine','PF-562271','Linsitinib',
                            'Paclitaxel','T0070907','VER155008','Erlotinib',
                            'LGK-974','Ilomastat','Ac-Gly-BoroPro','DMSO')

cluster_data$df_plot <- cluster_data$df %>%
  ungroup() %>%
  mutate(leiden_ordered = factor(as.integer(leiden), levels = order_leiden),
         leiden = factor(as.integer(leiden)),
         conc = factor(conc, levels = c('1_µM', '5_µM', '10_µM') %>% rev(),
                       labels = cluster_data$df$conc %>%
                         unique() %>%
                         rev() %>%
                         str_replace('_', ' ')),
         timepoint = factor(timepoint, labels = cluster_data$df$timepoint %>%
           unique() %>%
           str_to_sentence()  %>%
           str_replace('_', ' ')),
         compound = factor(compound, levels = order_compounds),
         compound_fig = factor(compound, levels = order_compounds_figure))

# plot leiden cluster distribution by conditions
p6 <- ggplot(cluster_data$df_plot, aes(x = leiden_ordered, y = compound)) +
  geom_point(pch = 21, col = 'black', aes(size = frac, fill = frac)) +
  facet_grid(conc ~ timepoint) +
  scale_fill_viridis_c(option = 'magma') +
  theme_bw() +
  theme(axis.title.y = element_blank())

# as barplot
p7 <- ggplot(cluster_data$df_plot, aes(x = compound_fig, y = frac, fill=leiden)) +
   geom_bar(stat='identity', col='grey70', linewidth=0.1, width=1) +
   facet_grid(conc ~ timepoint) +
   scale_fill_brewer(palette='Set3') +
   theme_bw() +
   scale_x_discrete(expand = c(0,0)) +
   scale_y_continuous(expand = c(0,0)) +
   theme(axis.title.x = element_blank(),
         axis.text.x = element_text(angle=90, vjust=0.5, hjust=1),
         axis.title.y = element_blank(),
         axis.text.y = element_blank(),
         axis.ticks.y = element_blank(),
         panel.grid.major = element_blank(),
         panel.grid.minor = element_blank(),
         panel.spacing = unit(.1,'lines'),
         text=element_text(size=6),
         strip.background = element_blank())

ggsave('data/processed/inhibitors/plots/organoid_embeddings_cluster_composition.pdf',
       plot = p7,
       dpi = 72, unit = 'mm', width = 180, height=40)

# merge figure
p_umaps <- p1 / p2 / p3 / p4 / p5
p <- (p_images | p_umaps | p6) + plot_layout(widths = c(2.5, 1, 2.5))
ggsave('data/processed/inhibitors/plots/organoid_embeddings.pdf', plot = p,
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)

# mahalanobis distances
list_mahal <- prepare_data_mahal(mdata_org)
list_mahal$mahal_results <- pairwise.mahalanobis(list_mahal$df_features %>%
  select(starts_with('pca')) %>%
  as.matrix(),list_mahal$df_features$condition)
list_plots <- arrange_mahal_plots(list_mahal, order_compounds)
p_mahal_dmso <- plot_mahalanobis_dmso(list_plots$plots)

# correlagramm for features
df_features_org_corr <- mdata_org['phenocoder_combined']$X %>% as_tibble()
df_features_org_corr <- df_features_org_corr %>% select(colnames(df_features_org_corr)[!duplicated(as.list(df_features_org_corr))])
# run correlation analysis
df_corr <- cor(df_features_org_corr, method = 'spearman')
df_corr_clean <- df_corr
df_corr_clean[is.na(df_corr_clean) | is.nan(df_corr_clean) | is.infinite(df_corr_clean)] <- 0
# build graph
construct_graph <- function(df, threshold) {
  # Set values below threshold to 0, above to 1 (adjacency matrix)
  adj_mat <- as.matrix(df)
  adj_mat[abs(adj_mat) < threshold] <- 0
  adj_mat[abs(adj_mat) >= threshold] <- 1
  diag(adj_mat) <- 0
  return(adj_mat)
}
A_corr <- construct_graph(df_corr_clean,0.5)
# plot with ggraph
library(ggraph)
library(igraph)
g <- graph_from_adjacency_matrix(A_corr, mode = 'undirected', weighted = TRUE, diag = FALSE)
p <- ggraph(g, layout = 'fr') +
  geom_edge_link(aes(edge_alpha = abs(weight), edge_width = abs(weight)), color = 'grey70') +
  geom_node_point(size = 2, color = 'black') +
  #geom_node_text(aes(label = name), repel = TRUE, size = 2.5) +
  theme_void() +
  theme(legend.position = 'none')
ggsave('data/processed/inhibitors/plots/correlation_graph_features.png', plot = p,
       dpi = 112, unit = 'mm', width = 400.05, height = 400.05)

bind_cols(list_mahal$df_features,mdata_org['phenocoder_combined']$X %>% as_tibble()) %>% left_join(list_mahal$mahal_results$distance['DMSO',] %>% enframe(name='condition', value='distance_dmso')) -> df_features_org_all

p1 <- ggplot(df_features_org_all %>% filter(condition != 'DMSO'), aes(x=log1p(distance_dmso))) + geom_histogram()
p2 <- ggplot(df_features_org_all %>% filter(condition != 'DMSO'), aes(x=log1p(distance_dmso), y=phenocoder_stat_area_chull_source)) + geom_point() + geom_smooth(method = 'lm')

# plot heatmap
library(pheatmap)
plot = pheatmap(as.matrix(df_corr_clean), show_rownames = FALSE, show_colnames = FALSE, treeheight_row=0, treeheight_col = 0, silent = TRUE, cellheight = 1, cellwidth = 1)
# save with ggsave
ggsave('data/processed/inhibitors/plots/correlation_heatmap_features.png', plot = plot[[4]],
       dpi = 112, unit = 'mm', width = 400.05, height = 400.05)


# lda analysis for positive ctrl vs neg ctrl
df_lda <- list_mahal$df_features
df_neg_ctrls <- df_lda %>% filter(compound == 'DMSO')
df_pos_ctrl <- df_lda %>% filter(condition %in% c('SN38_5 µM_Day 4','Bortezomib_5 µM_Day 4'))
df_lda <- bind_rows(df_neg_ctrls,df_pos_ctrl) %>% select(condition,pca_1:pca_32)
lda_result <- MASS::lda(condition ~ .,
  data=df_lda)
df_lda <- list_mahal$df_features %>% bind_cols(predict(object=lda_result, newdata = list_mahal$df_features)$x)
df_plot_lda <- df_lda %>% filter(condition %in% c('SN38_5 µM_Day 4','Bortezomib_5 µM_Day 4','DMSO'))

get_z_prime <- function(df_lda){
  df_z_prime <- df_lda %>%
  group_by(condition, compound, timepoint, conc) %>%
  summarise(mean = mean(LD1), sd = sd(LD1)) %>% mutate(mean_dmso = mean(df_lda %>% filter(condition == 'DMSO') |> pull(LD1)),
               sd_dmso = sd(df_lda %>% filter(condition == 'DMSO') |> pull(LD1)),
               z_prime = 1 - (3 * (sd_dmso + sd)) / abs(mean_dmso - mean))
  return(df_z_prime)
}

df_z_prime <- get_z_prime(df_lda)
df_plot_z_prime <- df_z_prime %>%
  filter(compound %in% c('Bortezomib','SN38', 'Trametinib')) %>%
  mutate(z_prime_categorical = ifelse(z_prime < 0, "z' < 0", ifelse(z_prime > 0.5, "z' > 0.5", "0 < z'< 0.5")),
         z_prime_categorical = factor(z_prime_categorical, levels = c("z' < 0","0 < z'< 0.5","z' > 0.5")),
         z_prime_round = round(z_prime, digits=2),
         timepoint = factor(timepoint, levels = c('Day 4', 'Day 7', 'Day 11')))

p_z_prime <- ggplot(df_plot_z_prime, aes(x=timepoint, y=conc, fill=z_prime_categorical)) +
  geom_tile(col='black', linewidth=1) +
  geom_text(aes(label=z_prime_round), size=6) +
  coord_fixed() +
  facet_wrap(~compound, nrow = 3) +
  theme_minimal() +
  xlab(NULL) +
  ylab(NULL) +
  scale_fill_grey(start=0.4, end=0.99) +
  theme(text=element_text(size=16),
         axis.ticks.y = element_blank(),
   axis.ticks.x = element_blank())

ggsave('data/processed/inhibitors/plots/z_prime.pdf', plot = p_z_prime,
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)

# interaction terms
celltypes_source <- list(cancer=c(2,3), caf=c(0,4,1,5))
celltypes_target <- list(cancer=c(3,4), caf=c(1,5,0,2,6))
celltypes <- list(source=celltypes_source, target=celltypes_target)
proliferation_annotation <- list(feature_type='target',cluster='4')
lamc2_annotation <- list(feature_type='source',cluster='2')
list_scores <- get_scores(mdata_org, celltypes, proliferation_annotation, lamc2_annotation)

p_heatmap_target <- plot_interaction_heatmap(list_scores, 'target', df_features = list_mahal$df_features)
p_heatmap_source <- plot_interaction_heatmap(list_scores, 'source', df_features = list_mahal$df_features)
p_heatmap_merged <-  plot_interaction_heatmap(list_scores, 'merged',
                         df_features = list_mahal$df_features,
                         list_df_de = df_de_conditions,
                         list_mahal)

ggsave('data/processed/inhibitors/plots/interaction_heatmap_merged.pdf', plot = p_heatmap_merged[[4]],
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)
