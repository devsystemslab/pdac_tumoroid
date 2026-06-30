# source utils
source('analysis/pilotscreen/utils.R')
# load libraries
library(ggridges)
library(RColorBrewer)
library(ggraph)
library(tidygraph)
library(ggrepel)
library(scales)
# set directories
dir_screen <- 'data/processed/tumoroidscreen'
dir_adata <- str_c(dir_screen, 'anndata', sep='/')
dir_plots <- str_c(dir_screen, 'plots', sep='/')
dir_analysis <- 'analysis/tumoroidscreen'
dir_metafiles <- 'metafiles'

# read mdata
mdata_org <- read_mdata(str_c(dir_adata, 'mdata_org_combined.h5mu', sep = '/'))
# source
mdata_cycle_01 <- read_mdata(str_c(dir_adata, 'mdata_cycle-01.h5mu', sep = '/'))
# target
mdata_cycle_03 <- read_mdata(str_c(dir_adata, 'mdata_cycle-03.h5mu', sep = '/'))

list_cycle_01 <- prepare_data_for_plotting(mdata_cycle_01)

list_cycle_03 <- prepare_data_for_plotting(mdata_cycle_03)


p_heatmap_01 <- pheatmap::pheatmap(list_cycle_01$M_phenocoder,
                                    scale = 'column',
                                    cellheight = 12,
                                    cellwidth = 12, silent=TRUE)[[4]]

p_heatmap_03 <- pheatmap::pheatmap(list_cycle_03$M_phenocoder,
                   scale = 'column',
                   cellheight = 12,
                    cellwidth = 12, silent=TRUE)[[4]]

# spatial plots for HM004
p_spatial_01 <- ggplot(list_cycle_01$phenocoder %>% filter(well_id == 'A08',plate_id=='HM004'), aes(x=`centroid-1`, y=`centroid-0`, col=leiden)) +
  geom_point() +
  coord_equal() +
  scale_color_brewer(palette = 'Set3')

grid.arrange(grobs=list(p_heatmap_01 ,p_spatial_01))


p_spatial_03 <- ggplot(list_cycle_03$phenocoder %>% filter(well_id == 'A08',plate_id=='HM004'), aes(x=`centroid-1`, y=`centroid-0`, col=leiden)) +
  geom_point() +
  coord_equal() +
  scale_color_brewer(palette = 'Set3')


grid.arrange(grobs=list(p_heatmap_03 ,p_spatial_03))

# interaction terms
celltypes_source <- list(cancer=c(0,3,4,5,6,7), caf=c(1,8,2))
celltypes_target <- list(cancer=c(0,1,2,3), caf=c(5,4))
celltypes <- list(source=celltypes_source, target=celltypes_target)
proliferation_annotation <- list(feature_type='target',cluster='2')
lamc2_annotation <- list(feature_type='source',cluster='6')
list_scores <- get_scores(mdata_org, celltypes, proliferation_annotation, lamc2_annotation, duct_scores = FALSE)
# remove NULL entries from list
list_scores[['target']] <- list_scores[['target']][!sapply(list_scores[['target']], is.null)]
list_scores[['source']] <- list_scores[['source']][!sapply(list_scores[['source']], is.null)]
plot_interaction_heatmap(list_scores, 'target', df_features = list_mahal$df_features)
plot_interaction_heatmap(list_scores, 'source', df_features = list_mahal$df_features)
plot_interaction_heatmap(list_scores, 'merged',
                         df_features = list_mahal$df_features,
                         list_df_de = df_de_conditions,
                         list_mahal)
feature_type = 'target'
df_scores <- reduce(list_scores[[feature_type]], left_join, by = c('id','leiden'))
colnames(df_scores) <- str_replace(colnames(df_scores), '\\.x', '_abs') %>%
      str_replace('\\.y', '_norm')
feature_cols <- df_scores %>% select(caf:area_chull) %>% colnames()
df_scores <-  df_scores  %>%
      mutate_at(feature_cols, rescale_quantile_clip)
df_scores <-  df_scores %>% left_join(list_mahal$df_features %>%
                                         mutate(id = str_c(well_id, plate_id,sep='_')) %>%
                                         select(id, CODENAME,GeneSymbol)) %>%
  mutate(CODENAME=ifelse(is.na(CODENAME),'DMSO',CODENAME))

list_mahal$mahal_results$distance['DMSO',] %>%
  enframe(name = 'compound', value = 'distance_dmso') %>%
  #filter(compound != 'DMSO') %>%
  mutate(compound_factor = factor(compound, levels = compound[order(distance_dmso)]),
         compound_int = as.integer(compound_factor)) -> df_dmso_mahal

df_scores_agg <- df_scores %>% group_by(CODENAME) %>% summarise_at(feature_cols, mean) %>%
  mutate_at(feature_cols, rescale_quantile_clip) %>%
  left_join(df_scores %>% select(CODENAME, GeneSymbol) %>% distinct()) %>%
  left_join(df_dmso_mahal %>% rename(CODENAME=compound) %>% select(CODENAME, distance_dmso) %>%
              mutate(distance_dmso=log1p(distance_dmso))) %>%
  arrange(-distance_dmso)

# filter for drugs that are in the top 100 genes
df_scores_agg <- map(top_genes%>%pull(gene), function(x){
  df_scores_agg %>% filter(str_detect(GeneSymbol,x) | distance_dmso > 8.5)
}) %>% bind_rows() %>% distinct()

# merge kernel sizes together
df_scores_agg <- df_scores_agg %>%
  rowwise() %>%
  mutate('caf-caf_abs' = mean(c_across(starts_with('radius')& ends_with('caf-caf_abs'))),
         'caf-caf_norm'= mean(c_across(starts_with('radius')& ends_with('caf-caf_norm'))),
         'cancer-cancer_abs' = mean(c_across(starts_with('radius')& ends_with('cancer-cancer_abs'))),
         'cancer-cancer_norm' = mean(c_across(starts_with('radius')& ends_with('cancer-cancer_norm'))),
         'cancer-caf_abs' = mean(c_across(starts_with('radius')& ends_with('cancer-caf_abs'))),
         'cancer-caf_norm' =mean(c_across(starts_with('radius')& ends_with('cancer-caf_norm')))
  )

feature_cols <- feature_cols[!str_detect(feature_cols,'radius')]
agg_cols <- colnames(df_scores_agg)[str_detect(colnames(df_scores_agg),'^caf-|^cancer-')]
M <- df_scores_agg %>%
    select(feature_cols, agg_cols) %>%
    as.matrix()


# M <- df_scores_agg %>% select(-CODENAME, -GeneSymbol, -distance_dmso) %>% as.matrix()
rownames(M) <- df_scores_agg$CODENAME
pal=brewer_pal(palette = 'Set3')
annotations_col <-  df_scores_agg %>% select(distance_dmso) %>% mutate(hit = ifelse(distance_dmso>8.5,1,0)) %>% as.data.frame()
rownames(annotations_col) <- df_scores_agg$CODENAME
p_heatmap_org_features <- pheatmap::pheatmap(t(M), scale='row',
                                             annotation_col = annotations_col,
                                             annotation_colors = list(hit=colorRampPalette(brewer.pal(9,"Greys"))(9),
                                                                      distance_dmso=colorRampPalette(brewer.pal(9,"Greys"))(9)),
                                             cellwidth = 15,
                                             cellheight = 15,
                                             labels_col = str_replace_na(df_scores_agg$CODENAME))

# save plot
ggsave('data/processed/tumoroidscreen/plots/interaction_heatmap_merged.pdf', plot = p_heatmap_org_features[[4]],
       dpi = 112, unit = 'mm', width = 400.05, height = 215.22)
# write to csv
df_scores_agg %>%
  arrange(-distance_dmso) %>%
  select(CODENAME, GeneSymbol, distance_dmso) %>%
  mutate(hit = ifelse(distance_dmso>8.5,1,0)) %>%
  left_join(tibble(CODENAME=p_heatmap_org_features$tree_col$labels, order=p_heatmap_org_features$tree_col$order)) %>%
  write.csv(str_c(dir_analysis,'tables','cpds_top100_genes_sorted_w_hits.csv',sep='/'))

df_scores_agg %>% select(GeneSymbol) %>%
  na.omit() %>% pull() %>%
  str_split(pattern = ';') %>%
  enframe(value = 'GeneSymbol') %>%
  unnest(cols=colnames(.)) %>%
  select(-name) %>% distinct() %>%
  write_tsv(str_c(dir_analysis,'tables','genes_distance_dmso_union_top100.tsv',sep='/'))






# extract organoid embedding from mdata
df_plot_umap <- mdata_org['phenocoder_combined']$obs %>%
as_tibble() %>%
bind_cols(mdata_org['phenocoder_combined']$obsm$X_umap %>%
        as_tibble() %>%
        rename(UMAP1 = V1, UMAP2 = V2))  %>% left_join(df_drugs) %>%
  mutate(compound = ifelse(negative_control == 'True', 'DMSO', CODENAME))
p_umap <- ggplot(df_plot_umap, aes(x=UMAP1,y=UMAP2, col=leiden)) +
  coord_equal() +
  theme_void() +
  scale_color_brewer(palette = 'Set3')
grid.arrange(grobs = list(p_heatmap_org_features,p_umap))
