library(tidyverse)
library(pheatmap)
library(readxl)
library(RColorBrewer)
library(scales)
# set working directory
dir_tables <- 'whole_mount_tumoroid/analysis/tumoroidscreen/tables/'
# read data
df_cpds <- read_csv(str_c(dir_tables,'top100_genes_sorted.csv')) %>%
  select(-`...1`)
df_de_caf <- read_excel(str_c(dir_tables,'pdac_caf_degenes_wc2.xlsx'))
df_de_cancer <- read_excel(str_c(dir_tables,'pdac_cancer_degenes_wc2.xlsx'))
df_expr_caf <- read_tsv(str_c(dir_tables,'pdac_caf_degenes_mean_exp_level1.txt'))
df_expr_cancer <- read_tsv(str_c(dir_tables,'pdac_cancer_degenes_mean_exp_level1.txt'))
# harmonize gene column name
colnames(df_expr_caf)[1] <- 'gene'
colnames(df_expr_cancer)[1] <- 'gene'
colnames(df_de_cancer) <- str_c('cancer_', colnames(df_de_cancer))
colnames(df_de_cancer)[1] <- 'gene'
colnames(df_de_caf) <- str_c('caf_', colnames(df_de_caf))
colnames(df_de_caf)[1] <- 'gene'

# define top gene targets
genes <- c('LPAR1', 'HDAC4', 'MT-CO1', 'DCK', 'TP53',
           'MTOR', 'CHEK2', 'CSNK2B', 'CSNK2A2', 'RRM1',
           'RPS6KB1', 'PDGFB', 'TBK1', 'PIK3CD', 'PIK3R2', 'PIK3C2B',
           'PAK4', 'MAP2K2', 'MAP2K5', 'MAP2K6', 'PRKCG', 'PRKCG',
           'TUBA1A', 'DYRK1B', 'TUBD1', 'PIKFYVE', 'BRAF')

# merge datasets
df <- left_join(df_expr_caf, df_expr_cancer) %>%
  left_join(df_de_cancer) %>%
  left_join(df_de_caf)

df_filtered <- df %>%
  left_join(df_cpds %>%
              filter(gene %in% genes) %>%
              select(gene, distance_sum_norm, top_100)) %>%
  filter(top_100 == 1)

df <- df %>%
  left_join(df_cpds %>%
              select(gene, distance_sum_norm, top_100)) %>%
  filter(top_100 == 1)

# plot expression heatmap for top cpd targets
cell_types <- df_filtered %>% select(PreCAF:basal) %>% colnames()
M <- df_filtered %>% select(cell_types) %>% as.matrix()
rownames(M) <- df_filtered %>% select(gene) %>% pull()
annotations_col <- df_filtered %>% select("cancer_-log10p") %>% as.data.frame()
rownames(annotations_col) <- df_filtered %>% select(gene) %>% pull()
p <- pheatmap(M, scale = 'row',
        # annotation_row = annotations_col,
        # annotation_colors = list(distance_sum_norm = colorRampPalette(brewer.pal(9, "Greys"))(9)),
        fontsize_col = 5, fontsize_row = 5, cellheight = 10, cellwidth = 10)
ggsave(plot=p[[4]],'whole_mount_tumoroid/analysis/tumoroidscreen/plots/heatmap_expr.pdf')

load(str_c(dir_tables,'tumoroidscreen.RData'))
# interaction terms
celltypes_source <- list(cancer=c(0,3,4,5,6,7), caf=c(1,8,2))
celltypes_target <- list(cancer=c(0,1,2,3), caf=c(5,4))
celltypes <- list(source=celltypes_source, target=celltypes_target)
proliferation_annotation <- list(feature_type='target',cluster='2')
lamc2_annotation <- list(feature_type='source',cluster='6')

for (feature_type in c('source', 'target')) {
  df_scores <- reduce(list_scores[[feature_type]], left_join, by = c('id', 'leiden'))
  colnames(df_scores) <- str_replace(colnames(df_scores), '\\.x', '_abs') %>%
    str_replace('\\.y', '_norm')
  feature_cols <- df_scores %>%
    select(caf:area_chull) %>%
    colnames()
  df_scores <- df_scores %>%
    mutate_at(feature_cols, rescale_quantile_clip)
  df_scores <- df_scores %>%
    left_join(list_mahal$df_features %>%
                mutate(id = str_c(well_id, plate_id, sep = '_')) %>%
                select(id, CODENAME, GeneSymbol)) %>%
    mutate(CODENAME = ifelse(is.na(CODENAME), 'DMSO', CODENAME))
  list_scores[['df']][[feature_type]] <- df_scores
}
cols_inter <- intersect(colnames(list_scores$df$source), colnames(list_scores$df$target))

list_mahal$mahal_results$distance['DMSO',] %>%
  enframe(name = 'compound', value = 'distance_dmso') %>%
  #filter(compound != 'DMSO') %>%
  mutate(compound_factor = factor(compound, levels = compound[order(distance_dmso)]),
         compound_int = as.integer(compound_factor)) -> df_dmso_mahal

list_scores[['df']][['agg']] <- list_scores$df$source %>% select(all_of(cols_inter), -leiden) %>%
  bind_rows(list_scores$df$target %>% select(all_of(cols_inter), -leiden)) %>%
    group_by(CODENAME, GeneSymbol) %>% summarise(across(where(is.numeric), mean)) %>%
   # mutate(across(where(is.numeric), rescale_quantile_clip)) %>%
  left_join(df_dmso_mahal %>% rename(CODENAME=compound) %>% select(CODENAME, distance_dmso) %>%
              mutate(distance_dmso=log1p(distance_dmso))) %>%
  arrange(-distance_dmso)




df_scores_agg <- list_scores$df$agg
# filter for drugs that are in the top 100 genes
df_scores_agg <- map(top_genes%>%pull(gene), function(x){
  df_scores_agg %>% filter(str_detect(GeneSymbol,x) | distance_dmso > 8.5)
}) %>% bind_rows() %>% distinct()
# merge kernel sizes together
df_scores_agg <- df_scores_agg %>%
  rowwise() %>%
  mutate('caf-caf_abs' = mean(c_across(starts_with('radius')& ends_with('caf-caf_abs')),na.rm = TRUE),
         'caf-caf_norm'= mean(c_across(starts_with('radius')& ends_with('caf-caf_norm')), na.rm = TRUE),
         'cancer-cancer_abs' = mean(c_across(starts_with('radius')& ends_with('cancer-cancer_abs')), na.rm = TRUE),
         'cancer-cancer_norm' = mean(c_across(starts_with('radius')& ends_with('cancer-cancer_norm')), na.rm = TRUE),
         'cancer-caf_abs' = mean(c_across(starts_with('radius')& ends_with('cancer-caf_abs')), na.rm = TRUE),
         'cancer-caf_norm' =mean(c_across(starts_with('radius')& ends_with('cancer-caf_norm')), na.rm = TRUE)
  )
# aggregated target expression
genes_intersect <- intersect(colnames(df_gene_matrix)[-1], df$gene)
df_expr_filtered <- df %>% filter(gene %in% genes_intersect)
M_expr <- df_expr_filtered %>% select(PreCAF:basal) %>% as.matrix()
# z score per gene
rownames(M_expr) <- df_expr_filtered$gene
M_dot <- gene_matrix[df_scores_agg$CODENAME, genes_intersect] %*% M_expr
M_dot <- t(scale(t(M_dot)))
df_annotation <- as_tibble(M_dot, rownames = 'CODENAME') %>% na.omit()
df_scores_agg <-  df_scores_agg %>% filter(CODENAME %in% df_annotation$CODENAME)



feature_cols <- feature_cols[!str_detect(feature_cols,'radius')]
agg_cols <- colnames(df_scores_agg)[str_detect(colnames(df_scores_agg),'^caf-|^cancer-')]
M <- df_scores_agg %>% ungroup() %>%
    select(c("caf",'cancer','cancer-caf_norm',
      "volume_chull","area_chull","caf-caf_abs","caf-caf_norm" ,"cancer-cancer_abs","cancer-cancer_norm","cancer-caf_abs")) %>%
    as.matrix()
rownames(M) <- df_scores_agg$CODENAME
pal=brewer_pal(palette = 'Set3')
annotations_col <-  df_scores_agg %>% select(distance_dmso) %>% as.data.frame()
rownames(annotations_col) <- df_scores_agg$CODENAME

p_heatmap_org_features <- pheatmap::pheatmap(t(M), scale='row',
                                             annotation_col = annotations_col %>% select(-CODENAME),
                                             cellwidth = 15,
                                             cellheight = 15,
                                             labels_col = str_replace_na(df_scores_agg$CODENAME))

M <- df_annotation %>% select(-CODENAME) %>% as.matrix()
rownames(M) <- df_annotation$CODENAME %>% str_remove('ROLIB')
M <- t(M)
M <- M[,p_heatmap_org_features$tree_col$labels[p_heatmap_org_features$tree_col$order]]
p_heatmap_target_expr <- pheatmap::pheatmap(M, scale='row',
                                             cluster_cols = FALSE,
                                             cellwidth = 15,
                                             cellheight = 15,
                                             labels_col = str_replace_na(colnames(M)))
# save plot
ggsave('whole_mount_tumoroid/analysis/tumoroidscreen/plots/interaction_heatmap_merged.pdf', plot = p_heatmap_org_features[[4]],
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)

ggsave('whole_mount_tumoroid/analysis/tumoroidscreen/plots/expr_heatmap_merged.pdf', plot = p_heatmap_target_expr[[4]],
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)






