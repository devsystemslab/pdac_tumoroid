library(tidyverse)
library(patchwork)
library(pheatmap)
library(FBN)
# define feature aggregation functions
get_celltype_interaction_scores <- function(df, feature_set, type, celltypes) {
  if (type == 'norm') {
    df <- df %>%
      select(dpt_pseudotime, plate,
             starts_with(str_c(feature_set, 'stat_interaction', sep = '_')) & contains(type))
  } else {
    df <- df %>%
      select(dpt_pseudotime, plate,
             starts_with(str_c(feature_set, 'stat_interaction', sep = '_')) & !contains('norm'))
    colnames(df) <- colnames(df) %>% str_replace('interaction_', 'interaction_abs_')
  }
  df <- df %>%
    pivot_longer(cols = starts_with(str_c(feature_set, 'stat_interaction', sep = '_')),
                 names_to = 'type', values_to = 'value') %>%
    separate(col = type, into = c('prefix', 'stat', 'interaction', 'type', 'from', 'to', 'radius'), sep = '_') %>%
    mutate(from = as.integer(from), to = as.integer(to),
           from = ifelse(from %in% celltypes$cancer, 'cancer', 'caf'),
           to = ifelse(to %in% celltypes$cancer, 'cancer', 'caf'),
           interaction_type = (ifelse(from == 'cancer' & to == 'cancer', 'cancer-cancer',
                                      ifelse(from == 'caf' & to == 'caf', 'caf-caf', 'cancer-caf')))) %>%
    group_by(interaction_type, dpt_pseudotime, radius) %>%
    summarise(score = mean(value)) %>%
    mutate(radius = as.integer(radius)) %>%
    ungroup()
  return(df)
}

get_spatial_corr_stats <- function(df, feature_set) {
  # interactions
  df_corr <- df %>%
    select(dpt_pseudotime,
           starts_with(feature_set) &
             contains('MoranI')) %>%
    arrange(dpt_pseudotime)
  colnames(df_corr) <- str_remove(colnames(df_corr), paste0(feature_set, '_', 'stat_moranI_'))
  df_corr <- df_corr %>%
    pivot_longer(cols = colnames(df_corr)[-1], names_to = 'feature') %>%
    separate(col = feature, into = c('feature', 'radius'), sep = '_') %>%
    mutate(radius = as.integer(radius))
  return(df_corr)
}

get_count_stats <- function(df, feature_set, clusters) {
  df_counts <- df %>%
    select(dpt_pseudotime, cell_count,
           str_c(feature_set, clusters[1], sep = '_'):str_c(feature_set, rev(clusters)[1], sep = '_')) %>%
    arrange(dpt_pseudotime)
  colnames(df_counts) <- str_remove(colnames(df_counts), paste0(feature_set, '_'))
  df_counts <- df_counts %>% pivot_longer(cols = colnames(df_counts)[-1], names_to = 'feature')
  return(df_counts)
}

get_size_stats <- function(df, feature_set) {
  df_size <- df %>%
    select(dpt_pseudotime, starts_with(feature_set) &
      contains('chull') &
      !contains('area')) %>%
    arrange(dpt_pseudotime)
  colnames(df_size) <- str_remove(colnames(df_size), paste0(feature_set, '_'))
  df_size <- df_size %>% pivot_longer(cols = colnames(df_size)[-1], names_to = 'feature')
  return(df_size)
}

get_centrality_scores <- function(df, feature_set) {
  df_centrality <- df %>%
    select(dpt_pseudotime,
           starts_with(feature_set) &
             contains('centrality')) %>%
    arrange(dpt_pseudotime)
  colnames(df_centrality) <- str_remove(colnames(df_centrality), paste0(feature_set, '_stat_centrality_'))
  df_centrality <- df_centrality %>%
    pivot_longer(cols = colnames(df_centrality)[-1], names_to = 'feature') %>%
    separate(col = feature, into = c('feature', 'stat', 'type', 'radius'), sep = '_') %>%
    mutate(radius = as.integer(radius)) %>%
    unite('feature', feature:type)
  return(df_centrality)
}

get_degree_scores <- function(df, feature_set) {
  df_degree <- df %>%
    select(dpt_pseudotime,
           starts_with(feature_set) &
             contains('stat_degree')) %>%
    arrange(dpt_pseudotime)
  colnames(df_degree) <- str_remove(colnames(df_degree), paste0(feature_set, '_stat_degree_'))
  df_degree <- df_degree %>%
    pivot_longer(cols = colnames(df_degree)[-1], names_to = 'feature') %>%
    separate(col = feature, into = c('feature', 'stat', 'radius'), sep = '_') %>%
    mutate(radius = as.integer(radius)) %>%
    unite('feature', feature:stat)
  return(df_degree)
}

get_all_scores <- function(df, feature_set = 'imp-neighbors', clusters, cell_annotations, duct_scores = FALSE) {
  df_counts <- get_count_stats(df, feature_set, clusters)
  df_size <- get_size_stats(df, feature_set)
  if (duct_scores) {
    df_duct_scores <- get_duct_scores(df, feature_set)
  } else {
    df_duct_scores <- NULL
  }
  df_interactions_norm <- get_celltype_interaction_scores(df, feature_set = feature_set,
                                                          type = 'norm', celltypes = cell_annotations)
  df_interactions_abs <- get_celltype_interaction_scores(df, feature_set = feature_set,
                                                         type = 'abs', celltypes = cell_annotations)
  df_corr <- get_spatial_corr_stats(df, feature_set)

  df_centrality <- get_centrality_scores(df, feature_set)

  df_degree <- get_degree_scores(df, feature_set)

  results <- list(
    df_counts = df_counts,
    df_size = df_size,
    df_interactions_abs = df_interactions_abs,
    df_interactions_norm = df_interactions_norm,
    df_duct_size = df_duct_scores,
    df_corr = df_corr,
    df_degree = df_degree,
    df_centrality = df_centrality)
  return(results)
}

load_features <- function(file, mods) {
  # read data
  df <- read_csv(file) %>% rename(index = `...1`)

  # unify column names
  for (i in names(mods)) {
    colnames(df) <- str_replace(colnames(df), i, mods[i])
  }
  return(df)
}

# load data and generate feature sets
files <- c(scaled='whole_mount_tumoroid/analysis/data/timecourse_features.csv',
           raw='whole_mount_tumoroid/analysis/data/timecourse_features_raw.csv')
# cluster annotation into caf and cancer
celltypes <- list(caf = c('3'),
                  cancer = c('0', '1', '2', '4'))
mods <- c('imputed_neighbors_bytimepoints_False' = 'imp-neighbors',
            'imputed_nuclei_bytimepoints_False' = 'imp-nuclei',
            'nuclei' = 'nuclei',
            'nuclei_msg' = 'nuclei-msg',
            'phenocoder' = 'phenocoder',
            'phenocoder_msg' = 'phenocoder-neighbors')
df <- load_features(files['scaled'],mods)
df_raw <- load_features(files['raw'],mods)
mods <- set_names(mods)
# get scores
list_features <- list(scaled=get_all_scores(df, feature_set = 'imp-neighbors',
                           clusters = as.character(0:4), cell_annotations = celltypes),
     raw=get_all_scores(df_raw, feature_set = 'imp-neighbors',
                           clusters = as.character(0:4), cell_annotations = celltypes))
list_features_dapi <- list(scaled=get_all_scores(df, feature_set = 'phenocoder',
                           clusters = as.character(0:4), cell_annotations = celltypes),
     raw=get_all_scores(df_raw, feature_set = 'phenocoder',
                           clusters = as.character(0:4), cell_annotations = celltypes))

# plotting
p1 <- ggplot(list_features$raw$df_interactions_abs %>%
               # group_by(dpt_pseudotime,interaction_type, score) %>%
               # summarise(score=mean(score)) ,
               filter(radius==150),
       aes(x = dpt_pseudotime, y = score, group = interaction_type, col=interaction_type)) +
  geom_smooth() +
  scale_color_manual(values = RColorBrewer::brewer.pal(5, 'Blues')[3:5] %>%
    set_names(list_features$scaled$df_interactions_norm %>%
                distinct(interaction_type) %>%
                pull() %>%
                sort() %>%
                as.factor())) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
   ylab("Interaction count") +
  xlab('DPT')

p2 <- ggplot(list_features$raw$df_interactions_abs %>%
               filter(radius==150) %>%
  left_join(list_features$raw$df_counts %>% filter(feature=='cell_count') %>% select(dpt_pseudotime,value)),
       aes(x = dpt_pseudotime, y = score/value, group = interaction_type, col=interaction_type)) +
  geom_smooth() +
  scale_color_manual(values = RColorBrewer::brewer.pal(5, 'Blues')[3:5] %>%
    set_names(list_features$scaled$df_interactions_norm %>%
                distinct(interaction_type) %>%
                pull() %>%
                sort() %>%
                as.factor())) +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
    ylab("Interaction Ratio") +
  xlab('DPT')


p3 <- list_features$raw$df_counts %>%
  ggplot(aes(x=dpt_pseudotime, y=value, group=feature, col=feature)) +
  geom_smooth() +
  scale_color_manual(values=c('cell_count'='grey30', '0'= '#1f77b4',
                              '1'='#ff7f0e', '2'='#2ca02c',
                              '3'= '#d62728', '4'='#9467bd')) +
  theme_bw() +
  ylim(c(0,NA)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
    ylab("Cell count") +
  xlab('DPT')

p4 <- list_features$raw$df_size %>% filter(feature=='stat_volume_chull') %>%
  ggplot(aes(x=dpt_pseudotime,y=value)) +
  geom_smooth(col='grey30') +
  theme_bw() +
  coord_cartesian(ylim = c(0, NA), xlim = c(0,1)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  ylab("Volume [mm^3]") +
  xlab('DPT')

  p5 <- list_features$raw$df_size %>% filter(feature=='stat_volume_chull') %>%
     mutate(percentile_low = quantile(value, .05),
         percentile_high = quantile(value, .95),
         value = ifelse(value < percentile_low, percentile_low, value),
         value = ifelse(value > percentile_high, percentile_high, value)) %>%
    left_join(list_features$raw$df_counts %>% filter(feature=='cell_count') %>%
     mutate(percentile_low = quantile(value, .05),
         percentile_high = quantile(value, .95),
         value = ifelse(value < percentile_low, percentile_low, value),
         value = ifelse(value > percentile_high, percentile_high, value)) %>% select(dpt_pseudotime,value) %>% rename(count=value)) %>%
  ggplot(aes(x=dpt_pseudotime,y=count/value)) +
  geom_smooth(col='grey30') +
  theme_bw() +
  coord_cartesian(ylim = c(0, NA), xlim = c(0,1)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  ylab("Density [mm^-3]") +
  xlab('DPT')


p <- p1 / (p3 | p4 |p5)
ggsave('whole_mount_tumoroid/analysis/timecourse/plots/dpt_tumoroid_features.pdf',p)

df_corr_if <- list_features$raw$df_corr %>% filter(radius == 150,
                                          !feature %in% c('0','1','2','3','4')) %>%
  group_by(feature) %>%
  mutate(percentile_low = quantile(value, .05),
         percentile_high = quantile(value, .95),
         value = ifelse(value < percentile_low, percentile_low, value),
         value = ifelse(value > percentile_high, percentile_high, value)) %>%
  pivot_wider(values_from = value, names_from = feature, id_cols = dpt_pseudotime) %>%
  arrange(dpt_pseudotime)

df_annotation <- df_corr_if %>% select(dpt_pseudotime) %>% left_join(df %>% select(dpt_pseudotime,plate)) %>% select(-dpt_pseudotime) %>% as.data.frame(row.names = as.character(rank(df_corr_if$dpt_pseudotime)))

M <- df_corr_if %>%
  select(-dpt_pseudotime) %>%
  mutate_all(meanFilter, 50) %>%
  as.matrix()
rownames(M) <- as.character(rank(df_corr_if$dpt_pseudotime))
# 1d mean filter
p5 <- pheatmap(t(M), cluster_cols = FALSE,
         cellheight = 10,
         scale = 'row',
         annotation_col = df_annotation,
         annotation_colors = list(plate=c('001'='#1964B0','002'='#00C992','003'='#F4A637', '004'='#DB5829','005'='#894B45')),
         show_colnames = FALSE, silent = FALSE)
ggsave(plot=p5[[4]],'sp_corr_heatmap_dpt.pdf')

df_corr_cl <- list_features$raw$df_corr %>% filter(#radius == 25,
                                          feature %in% c('0','1','2','3','4')) %>%
  group_by(feature) %>%
  mutate(percentile_low = quantile(value, .05),
         percentile_high = quantile(value, .95),
         value = ifelse(value < percentile_low, percentile_low, value),
         value = ifelse(value > percentile_high, percentile_high, value))

p6 <- df_corr_cl %>% filter(radius == 150) %>%
  ggplot(aes(x=dpt_pseudotime,y=value, group=feature, col=feature)) +
  geom_smooth() +
  theme_bw() +
  coord_cartesian(ylim = c(0, NA), xlim = c(0,1)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
   scale_color_manual(values=c('0'= '#1f77b4',
                              '1'='#ff7f0e', '2'='#2ca02c',
                              '3'= '#d62728', '4'='#9467bd')) +
  ylab("Moran's I") +
  xlab('DPT')
ggsave('sp_corr_cl_dpt.pdf')


p1 + p2 + p3 + p4 + p5 + p6 + plot_layout(guides='collect')

M_corr <- list_features$raw$df_corr %>%
  filter(radius == 150) %>%
  pivot_wider(values_from = value, names_from = feature, id_cols = dpt_pseudotime) %>%
  select(-dpt_pseudotime) %>% cor()
rows <- rownames(M_corr)[rownames(M_corr) %in% c('0','1','2','3','4')]
cols <- rownames(M_corr)[!rownames(M_corr) %in% c('0','1','2','3','4')]
p <- pheatmap(M_corr[rows,cols], cellheight = 20, cellwidth = 20, scale='row')
ggsave(plot=p[[4]],'whole_mount_tumoroid/analysis/timecourse/plots/sp_corr_if_cl_dpt.pdf')

M <- df_corr_cl %>%
  pivot_wider(values_from = value, names_from = feature, id_cols = dpt_pseudotime) %>%
  arrange(dpt_pseudotime) %>%
  select(-dpt_pseudotime) %>%
  mutate_all(meanFilter, 50) %>%
  as.matrix()

# 1d mean filter
p6 <- pheatmap(t(M), cluster_cols = FALSE, cellheight = 10, scale = 'row', silent = FALSE)
ggsave(plot=p6[[4]],'sp_corr_cl_heatmap_dpt.pdf')


# boxplots per timepoint
p <- df_raw  %>%
  select(index,plate_id,`imp-nuclei_stat_volume_chull`,cell_count) %>%
  ggplot(aes(x=plate_id,y=`imp-nuclei_stat_volume_chull`, fill=plate_id)) +
  geom_boxplot(outlier.alpha = 0) +
  scale_fill_manual(values=c('001'='#1964B0','002'='#00C992','003'='#F4A637', '004'='#DB5829','005'='#894B45')) +
  theme_bw() +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major = element_blank())

ggsave(plot=p,'boxplot_volume.pdf')
