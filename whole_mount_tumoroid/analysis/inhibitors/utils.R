library(tidyverse)
library(magick)
library(ggrepel)
library(gridExtra)
library(grid)
library(patchwork)
library(scales)

err_conda <- tryCatch({reticulate::use_condaenv('rapid_sc')},
                      error = 'fail',
                      {print('Initial reticulate loading attempt failed.')})
reticulate::use_condaenv('rapid_sc')
builtins <- reticulate::import_builtins()
err_anndata <- tryCatch({library(anndata)}, error = 'fail', {print('Initial anndata loading attempt failed.')})
library(anndata)
err_muon <- try({reticulate::import("muon")}, silent = TRUE)

prepare_data_mahal <- function(mdata_org){
  df_features <- mdata_org['phenocoder_combined']$obs %>%
  as_tibble() %>%
  bind_cols(as_tibble(mdata_org['phenocoder_combined']$obsm$X_pca) %>%
              rename(all_of(colnames(.) %>% set_names(str_c('pca_', 1:32))))) %>%
  mutate(compound = as.character(compound) %>% str_replace('_', ' '),
         conc = as.character(conc)%>% str_replace('_', ' '),
         timepoint = as.character(timepoint)%>%
           str_replace('_', ' ') %>%
           str_to_sentence()) %>%
  bind_cols(mdata_org['phenocoder_combined']$obsm$X_umap %>%
              as_tibble() %>%
              rename(UMAP1 = V1, UMAP2 = V2))  %>%
  mutate(compound = ifelse(conc == '0 µM', 'DMSO', compound),
         conc = ifelse(compound == 'DMSO', '0 µM', conc)) %>%
  mutate(timepoint = factor(timepoint, levels = c('Day 4', 'Day 7', 'Day 11')),
         conc = factor(conc, levels = c('0 µM', '1 µM', '5 µM', '10 µM')%>% rev())) %>%
  mutate(condition = str_c(compound, conc, timepoint, sep = '_'),
         condition = ifelse(compound == 'DMSO', compound, condition),
         timepoint = ifelse(compound == 'DMSO', 'Day 4', as.character(timepoint)))
  return(list(df_features=df_features))
}

pairwise.mahalanobis <- function(x, grouping = NULL, cov = NULL, inverted = FALSE, digits = 5, ...) {
  # standardize input data as matrix
  x <- if (is.vector(x)) {
    matrix(x, ncol = length(x))
  } else { as.matrix(x) }

  if (!is.matrix(x)) {
    stop("x could not be forced into a matrix")
  }
  # no group assigned, uses first col
  if (length(grouping) == 0) {
    grouping <- t(x[1])
    x <- x[2:dim(x)[2]]
    cat("assigning grouping\n")
    print(grouping)
  }
  # get dims
  n <- nrow(x)
  p <- ncol(x)

  # grouping and matrix do not correspond
  if (n != length(grouping)) {
    cat(paste("n: ", n, "and groups: ", length(grouping), "\n"))
    stop("nrow(x) and length(grouping) are different")
  }
  # groups
  g <- as.factor(grouping)
  # elements in each group
  lev <- lev1 <- levels(g)
  counts <- as.vector(table(g))

  # remove grouping if not represented in data
  if (any(counts == 0)) {
    empty <- lev[counts == 0]
    warning(sprintf(ngettext(length(empty), "group %s is empty",
                             "groups %s are empty"), paste(empty, collapse = " ")),
            domain = NA)
    lev1 <- lev[counts > 0]
    g <- factor(g, levels = lev1)
    counts <- as.vector(table(g))
  }

  ng <- length(lev1)
  # g x p matrix of group means from x
  group_means <- tapply(x, list(rep(g, p), col(x)), mean)

  # create covariance matrix, standardize into correlation mtx
  if (missing(cov)) {
    inverted <- FALSE
    cov <- cor(x)
  }
  else {
    # check cov of correct dimension
    if (dim(cov) != c(p, p))
      stop("cov matrix not of dim = (p,p)\n")
  }

  # initialize distance matrix
  distance <- matrix(nrow = ng, ncol = ng)
  dimnames(distance) <- list(rownames(group_means), rownames(group_means))

  means <- round(group_means, digits)
  cov <- round(cov, digits)
  distance <- round(distance, digits)

  for (i in 1:ng) {
    distance[i,] <- mahalanobis(group_means, group_means[i,], cov, inverted)
  }

  result <- list(means = group_means, cov = cov, distance = distance, counts = counts)
  return(result)
}

plot_mahalanobis_conc_time <- function(conc, time, M, order_compounds) {
  min_val <- min(M)
  max_val <- quantile(M, .99)
  breaks <- seq(min_val, max_val, length.out = 101)
  conditions <- rownames(M)
  selected <- str_detect(conditions, time) & str_detect(conditions, conc)
  selected_DMS0 <- str_detect(conditions, 'DMSO')
  selected <- selected | selected_DMS0
  M <- M[selected, selected]
  rownames(M) <- str_remove(rownames(M), str_c('_', conc)) %>% str_remove(str_c('_', time))
  colnames(M) <- rownames(M)
  compounds_missing <- order_compounds[!order_compounds %in% rownames(M)]
  for (compound in compounds_missing) {
    M <- cbind(M, c(NA) %>% set_names(compound))
    colnames(M)[length(colnames(M))] <- compound
    M <- rbind(M, c(NA) %>% set_names(compound))
    rownames(M)[length(rownames(M))] <- compound
  }
  M <- M[order_compounds, order_compounds]
  heatmap <- pheatmap::pheatmap(M, breaks = breaks, show_colnames = TRUE, angle_col = "315", cellheight = 10, cellwidth = 10,
                                cluster_cols = FALSE, cluster_rows = FALSE, gaps_row = 1, gaps_col = 1,
                                silent = TRUE, na_col = 'darkred')
  return(list(plot = heatmap[[4]], matrix = M))
}

arrange_mahal_plots <- function(list_mahal, order_compounds){
  df_iter <- list_mahal$df_features %>%
    distinct(conc, timepoint) %>%
    filter(conc != '0 µM')
  list_plots <- map2(as.character(df_iter$conc) %>% set_names(str_c(df_iter$conc, df_iter$timepoint, sep = ' ')),
                     df_iter$timepoint, plot_mahalanobis_conc_time,
                     M = list_mahal$mahal_results$distance,
                     order_compounds = order_compounds)
  grobs_with_titles <- lapply(seq_along(list_plots), function(i) {
    arrangeGrob(list_plots[[i]]$plot, top = textGrob(names(list_plots)[i], gp = gpar(fontsize = 14, fontface = "bold")))
  })
  p_mahal_conc_time <- grid.arrange(grobs = grobs_with_titles, ncol = 3)
  return(list(plots=list_plots, arranged_plot=p_mahal_conc_time))
}


plot_mahalanobis_dmso <- function(list_plots){
  list_dmso <- map(list_plots, function(x) {
    M <- x$matrix
    col_DMSO <- colnames(M) == 'DMSO'
    M <- M[,col_DMSO]
    return(M)
  })
  df_dmso_mahal <- bind_rows(list_dmso, .id = 'condition') %>%
    select(-DMSO) %>%
    mutate(condition = str_replace(condition, ' Day', '_Day')) %>%
    separate(condition, into = c('conc', 'timepoint'), sep = '_', remove = FALSE) %>%
    mutate(timepoint = str_remove(timepoint, 'Day ') %>% as.integer(),
           conc = str_remove(conc, ' µM') %>% as.integer()) %>%
    arrange(timepoint, -conc)

  annotation_row <- df_dmso_mahal %>%
    select(conc, timepoint) %>%
    as.data.frame()
  rownames(annotation_row) <- df_dmso_mahal$condition
  annotation_row$conc <- factor(annotation_row$conc)
  annotation_row$timepoint <- factor(annotation_row$timepoint)
  M_dmso_mahal <- df_dmso_mahal %>%
    select(-c('conc', 'timepoint', 'condition')) %>%
    as.matrix()
  rownames(M_dmso_mahal) <- df_dmso_mahal$condition

  compounds <- c('BTT-3033',
                 'Linsitinib',
                 'Erlotinib',
                 'Trametinib',
                 'T0070907',
                 'PF-562271',
                 'VER155008',
                 'Ilomastat',
                 'LGK-974',
                 'Ac-Gly-BoroPro',
                 'Gemcitabine',
                 'Paclitaxel',
                 'Bortezomib',
                 'SN38')
  target <- c('ITGA2/B1',
              'IGF-1R',
              'EGFR',
              'MEK',
              'PPARG',
              'FAK',
              'HSP70',
              'MMP7',
              'PORCN/WNT',
              'FAP',
              'CMPK1/TYMS',
              'TUBB1/BCL2',
              'PSMB5/PRSS1/PSMB1',
              'TOP1')
  pathways <- c('Integrin Signaling Pathway',
                'Insulin Signaling Pathway',
                'MAPK/ERK Signaling Pathway',
                'MAPK/ERK Signaling Pathway',
                'Adipogenesis Pathway',
                'Focal Adhesion Signaling Pathway',
                'Heat Shock Response Pathway',
                'ECM Remodeling Pathway',
                'Wnt Signaling Pathway',
                'Tissue Remodeling',
                'DNA Synthesis',
                'Microtubule Dynamics',
                'Ubiquitin-Proteasome Pathway',
                'Irinotecan Metabolism Pathway')

  annotation_col <- data.frame(target, pathways)
  rownames(annotation_col) <- compounds

  # Specify colors
  ann_colors <- list(
    timepoint = c("#ecf8fb", "#b2e1e2", "#2ba25f") %>% set_names(levels(annotation_row$timepoint)),
    conc = c("#ecf8fb", "#b2cde3", "#8856a7") %>% set_names(levels(annotation_row$conc)))

  p <- pheatmap::pheatmap(M_dmso_mahal %>% t(),
                          breaks = seq(min(M_dmso_mahal, na.rm = TRUE), quantile(M_dmso_mahal, .99, na.rm = TRUE), length.out = 101),
                          annotation_col = annotation_row,
                          #annotation_col = annotation_col,
                          annotation_colors = ann_colors,
                          gaps_col = c(3, 6),
                          #angle_col = "315",
                          show_rownames = TRUE,
                          show_colnames = FALSE,
                          cluster_cols = FALSE,
                          cluster_row = TRUE,
                          cellwidth = 25,
                          cellheight = 25,
                          na_col = 'darkred')
}


read_mdata <- function(file) {
  mu <- reticulate::import("muon")
  mdata <- reticulate::py_to_r(mu$read_h5mu(file))
  return(mdata)
}

get_data_from_mdata <- function(mdata, mod) {
  df <- mdata[mod]$obs %>%
    as_tibble() %>%
    select(z, `centroid-1`, `centroid-0`, leiden, well_id, plate_id) %>%
    bind_cols(as_tibble(mdata[mod]$X))
  if (!is.null(mdata[mod]$obsm$X_umap)) {
    df <- df %>%
      bind_cols(as_tibble(mdata[mod]$obsm$X_umap) %>% rename(UMAP1 = V1, UMAP2 = V2))
  }
  return(df)
}

prepare_data_for_plotting <- function(mdata, source_mod = 'nuclei', target_mod = 'phenocoder') {
  df_phenocoder <- get_data_from_mdata(mdata, target_mod)
  df_nuclei <- get_data_from_mdata(mdata, source_mod) %>% mutate(leiden_phenocoder = df_phenocoder$leiden)
  df_avg_phenocoder <- df_nuclei %>%
    group_by(leiden_phenocoder) %>%
    summarise_at(vars(mdata[source_mod]$var_names), mean)
  M_phenocoder <- df_avg_phenocoder %>%
    select(-leiden_phenocoder) %>%
    as.matrix()
  rownames(M_phenocoder) <- df_avg_phenocoder$leiden_phenocoder
  df_avg_nuclei <- df_nuclei %>%
    group_by(leiden) %>%
    summarise_at(vars(mdata[source_mod]$var_names), mean)
  M_nuclei <- df_avg_nuclei %>% select(-leiden) %>% as.matrix()
  rownames(M_nuclei) <- df_avg_nuclei$leiden
  results <- list(target_mod = df_phenocoder, source_mod = df_nuclei,
                  df_avg_phenocoder = df_avg_phenocoder, M_phenocoder = M_phenocoder,
                  df_avg_nuclei = df_avg_nuclei, M_nuclei = M_nuclei)
  names(results) <- c(target_mod, source_mod, 'df_avg_phenocoder', 'M_phenocoder', 'df_avg_nuclei', 'M_nuclei')
  return(results)
}

prepare_umap_org_data <- function(mdata_org) {
  df_plot_umap <- mdata_org['phenocoder_combined']$obs %>%
    as_tibble() %>%
    mutate(compound = as.character(compound) %>% str_replace('_', ' '),
           conc = as.character(conc) %>% str_replace('_', ' '),
           timepoint = as.character(timepoint) %>%
             str_replace('_', ' ') %>%
             str_to_sentence()) %>%
    bind_cols(mdata_org['phenocoder_combined']$obsm$X_umap %>%
                as_tibble() %>%
                rename(UMAP1 = V1, UMAP2 = V2)) %>%
    mutate(compound = ifelse(conc == '0 µM', 'DMSO', compound),
           conc = ifelse(compound == 'DMSO', '0 µM', conc)) %>%
    mutate(timepoint = factor(timepoint, levels = c('Day 4', 'Day 7', 'Day 11')),
           conc = factor(conc, levels = c('0 µM', '1 µM', '5 µM', '10 µM') %>% rev()))
  df_umap_dmso <- df_plot_umap %>% filter(compound == 'DMSO')
  df_plot_umap <- map(df_plot_umap$timepoint %>% unique(), function(x) {
    df_umap_dmso %>% mutate(timepoint = x)
  }) %>%
    bind_rows() %>%
    bind_rows(df_plot_umap %>% filter(compound != 'DMSO'))
  return(df_plot_umap)
}



prepare_cluster_data <- function(mdata_org){
   df <- mdata_org['phenocoder_combined']$obs %>%
    as_tibble() %>%
    select(-c('leiden_target','leiden_source')) %>%
    mutate_at(vars(c('compound', 'conc', 'leiden')), as.character) %>%
    mutate(compound = ifelse(conc == '0_µM', 'DMSO', compound))

  df_dmso <- df %>%
    filter(compound == 'DMSO') %>%
    mutate(n_total = n()) %>%
    group_by(leiden) %>%
    summarise(frac = n() / n_total) %>%
    distinct(leiden, frac)
  df_dmso <- df %>%
    filter(conc != '0_µM') %>%
    select(timepoint, conc) %>%
    distinct() %>%
    cross_join(df_dmso) %>%
    mutate(compound = 'DMSO')

  df <- df %>%
    filter(compound != 'DMSO') %>%
    group_by(compound, conc, timepoint) %>%
    mutate(n_total = n()) %>%
    group_by(compound, conc, timepoint, leiden) %>%
    mutate(n = n(), frac = n / n_total) %>%
    select(frac) %>%
    distinct() %>%
    bind_rows(df_dmso)
  return(list(df=df,df_dmso=df_dmso))
}

get_cluster_order <- function(cluster_data){
  order_leiden <- cluster_data$df_dmso %>%
                    distinct(leiden, frac) %>%
                    mutate(rank = rank(frac)) %>%
                    arrange(-rank) %>%
                    pull(leiden) %>% as.integer()
  all_leiden_clusters <- as.integer(distinct(ungroup(cluster_data$df),leiden) %>% pull())

  order_leiden <- c(order_leiden,all_leiden_clusters[!all_leiden_clusters %in% order_leiden])

  return(order_leiden)
}

get_compound_order <- function(cluster_data){
  df_dist <- cluster_data$df %>%
  group_by(compound, leiden) %>%
  summarise(frac = mean(frac)) %>%
  arrange(leiden) %>%
  pivot_wider(id_cols = compound, values_from = frac, names_from = leiden) %>%
  mutate_all(~replace_na(.x, 0)) %>%
  ungroup()
  M <- df_dist %>% select(-compound) %>% as.matrix()
  rownames(M) <- df_dist$compound
  filter_dmso <- rownames(M) == 'DMSO'
  M <- dist(M, diag = 1, upper = 1) %>% as.matrix()
  order_compounds <- M[which(filter_dmso),] %>% sort() %>% names()
  return(order_compounds)
}



plot_nuclei_data <- function(list_nuclei, list_msg, type = 'phenocoder', color_set = 'Set3', frac = 0.0001) {

  p_spatial <- ggplot(list_msg[[str_c(type, '_msg')]] %>% sample_frac(frac), aes(UMAP1, UMAP2, col = leiden)) +
    geom_point(size = 0.001) +
    theme_void() +
    coord_equal() +
    scale_color_brewer(palette = color_set) +
    guides(color = guide_legend(ncol = 1, override.aes = list(size = 3))) +
    labs(color = element_blank()) +
    theme(legend.position = 'left') +
    ggtitle(str_c('Spatial neighborhoods', type, sep = '-'))

  p_nuc <- ggplot(list_nuclei[[type]] %>% sample_frac(frac), aes(UMAP1, UMAP2, col = leiden)) +
    geom_point(size = 0.01) +
    theme_void() +
    coord_equal() +
    scale_color_brewer(palette = color_set) +
    guides(color = guide_legend(ncol = 1, override.aes = list(size = 3))) +
    labs(color = element_blank()) +
    theme(legend.position = 'right') +
    ggtitle(str_c('Nuclei', type, sep = '-'))

  p_pheat_spatial <- pheatmap::pheatmap(list_msg[[str_c('M', type, sep = '_')]],
                                        scale = 'column',
                                        cellheight = 12,
                                        cellwidth = 12)
  p_pheat_nuc <- pheatmap::pheatmap(list_nuclei[[str_c('M', type, sep = '_')]],
                                    scale = 'column',
                                    cellheight = 12,
                                    cellwidth = 12)

  p <- grid.arrange(grobs = list(p_spatial, p_nuc, p_pheat_spatial[[4]], p_pheat_nuc[[4]]))
  return(p)
}

prepare_data_de <- function(mdata_org, mod='phenocoder_combined'){
  df <- mdata_org[mod]$obs %>%
    as_tibble() %>%
    bind_cols(as_tibble(mdata_org[mod]$X)) %>%
    mutate(compound=ifelse(conc=='0_µM','DMSO', as.character(compound)))
  M <- df %>% select(cell_count_target:density_chull_sum_source) %>% as.matrix()
  rownames(M) <- df %>% mutate(id=str_c(well_id, plate_id, sep = '_')) %>% pull(id)
  df <- df %>% mutate(condition = str_c(as.character(conc),
                                  as.character(timepoint),
                                  as.character(compound), sep = '-'),
                condition = ifelse(compound=='DMSO','DMSO',condition))
  return(list(M=M, df_features = df))
}



run_presto <- function(M, groups, filter_groups=NULL) {
  # plot DE features
  df_de <- presto::wilcoxauc(t(M), groups, groups_use=filter_groups) %>%
    as_tibble() %>%
    mutate(rank = rank(padj)) %>%
    group_by(group) %>%
    mutate(rank_group = rank(rank),
           color = ifelse(abs(logFC) >= 1 & padj < 0.001, 'red', 'grey50'))
  # df_de$group <- factor(df_de$group, levels = unique(df_de$group) %>%
  #   as.integer() %>%
  #   sort() %>%
  #   as.character())
  return(df_de)
}

plot_de <- function(df_de) {
  selected_features <- df_de %>%
    filter(color == 'red') %>%
    distinct(feature) %>%
    pull()

  p1 <- ggplot(df_de, aes(x = logFC, y = -log10(padj), col = color)) +
    geom_point(size = .25) +
    facet_wrap(~group, ncol = 4) +
    theme_light() +
    geom_vline(xintercept = c(1, -1), linetype = "dashed", col = 'grey70') +
    geom_hline(yintercept = -log10(0.001), linetype = "dashed", col = 'grey70') +
    geom_text_repel(data = df_de %>% filter(color != 'grey50', rank_group < 6, feature %in% selected_features),
                    aes(label = feature), col = 'black', size = 2, box.padding = 0.1, max.overlaps = 20) +
    coord_cartesian(clip = "off") +
    scale_color_identity() +
    theme(aspect.ratio = 1) +
    ylab('-log10(padj)')
  return(p1)
}

plot_organoid_overlays <- function(df_example_images,
                                   well_id,
                                   plate_id,
                                   scale = NULL,
                                   scale_bar = FALSE,
                                   size = 1) {
  df_image <- df_example_images %>%
    filter(well_id == well_id, plate_id == plate_id) %>%
    mutate(compound = ifelse(conc == '0_µM', 'DMSO', compound),
           conc = ifelse(compound == 'DMSO', '', conc))
  image <- image_read(c(df_image$file_cycle_1, df_image$file_cycle_3)) %>% image_normalize()
  image <- image_append(image) %>% image_border("white", "4x4")

  width <- image_info(image) %>% pull(width)
  height <- image_info(image) %>% pull(height)
  if (!is.null(scale)) {
    width_scale <- round(width * scale)
    image <- image %>% image_scale(as.character(width_scale))
  }

  # set image as background in the plot
  bg <- rasterGrob(image, width = unit(1, 'npc'), height = unit(1, 'npc'), interpolate = TRUE)

  p1 <- ggplot(df_image) +
    coord_fixed(clip = 'off') +
    annotation_custom(grob = bg,
                      xmin = 0,
                      xmax = width,
                      ymin = 0,
                      ymax = height) +
    scale_x_continuous(limits = c(0, width),
                       expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, height),
                       expand = c(0, 0)) +
    theme_void() +
    geom_label(label = str_c(str_replace(df_image$timepoint, '_', ' ') %>% str_to_sentence(),
                             df_image$compound,
                             str_replace(df_image$conc, '_', ' '), sep = ' '),
               size = 1.5,
               x = 10, y = height - 50, hjust = 'inward',
               col = 'white',
               # fill = adata$uns$leiden_colors[(df_image$leiden) + 1],
               fill = 'grey70',
               fontface = "bold") +
    theme(legend.position = 'none')

  if (scale_bar) {
    p1 <- p1 +
      theme_void() +
      geom_rect(xmin = width - 100 - 615.384,
                xmax = width - 100,
                ymin = 150,
                ymax = 200,
                fill = 'white',
                col = 'white')
  }
  return(p1)
}


get_proliferation_scores <- function(mdata, feature_type, proliferative_cluster) {
  df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(c(str_c('cell_count', feature_type, sep = '_'),
             str_c('phenocoder', proliferative_cluster, feature_type, sep = '_'))) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden,
           id = rownames(mdata['phenocoder_combined']$obs)) %>%
    rename(cell_count = str_c('cell_count', feature_type, sep = '_'),
           proliferative_cluster = str_c('phenocoder', proliferative_cluster, feature_type, sep = '_')) %>%
    mutate(proliferative_frac = proliferative_cluster / cell_count)
  return(df)
}


get_lamc2_scores <- function(mdata, feature_type, lamc2_cluster) {
  if (length(lamc2_cluster) == 1) {
    df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(c(str_c('cell_count', feature_type, sep = '_'),
             str_c('phenocoder', lamc2_cluster, feature_type, sep = '_'))) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden,
           id = rownames(mdata['phenocoder_combined']$obs)) %>%
    rename(cell_count = str_c('cell_count', feature_type, sep = '_'),
           lamc2_cluster = str_c('phenocoder', lamc2_cluster, feature_type, sep = '_')) %>%
      mutate(lamc2_frac = lamc2_cluster / cell_count) %>%
    select(-cell_count)
  } else {
    df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(c(str_c('cell_count', feature_type, sep = '_'),
             str_c('phenocoder', lamc2_cluster, feature_type, sep = '_'))) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden,
           id = rownames(mdata['phenocoder_combined']$obs)) %>%
    rename(cell_count = str_c('cell_count', feature_type, sep = '_'),
           lamc2_cluster = str_c('phenocoder', lamc2_cluster, feature_type, sep = '_')) %>%
    rowwise() %>%
    mutate(lamc2_count = sum(c_across(cols=starts_with('lamc2')))) %>%
    mutate(lamc2_frac = lamc2_count / cell_count) %>%
    select(lamc2_count,lamc2_frac,leiden,id) %>%
    as_tibble()
  }
  return(df)
}

get_size_scores <- function(mdata, feature_type) {
  df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(contains('chull') &
             contains(feature_type) &
             starts_with('phenocoder_stat')) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden,
           id = rownames(mdata['phenocoder_combined']$obs))
  colnames(df) <- str_remove(colnames(df), 'phenocoder_stat_') %>% str_remove(str_c('_', feature_type))
  return(df)
}

get_duct_scores <- function(mdata, feature_type) {
  df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(str_c(c('volume_chull_mean',
                   'volume_chull_sum',
                   'n_pts_sum',
                   'n_pts_mean',
                   'density_chull_mean',
                   'density_chull_sum',
                   'distance_center_mean',
                   'distance_center_sum',
                   'n_chulls'),
                 feature_type, sep = '_')) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden,
           id = rownames(mdata['phenocoder_combined']$obs))
  colnames(df) <- str_remove(colnames(df), str_c('_', feature_type))
  return(df)
}

get_celltype_scores <- function(mdata, feature_type, celltypes) {
  df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(starts_with('phenocoder_'), -starts_with('phenocoder_msg')) %>%
    select(ends_with(feature_type)) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden,
           id = rownames(mdata['phenocoder_combined']$obs)) %>%
    select(-starts_with('phenocoder_stat')) %>%
    pivot_longer(cols = starts_with('phenocoder'), names_to = 'type', values_to = 'value') %>%
    separate(type, sep = '_', into = c('prefix', 'cluster', 'suffix')) %>%
    mutate(leiden = leiden, cluster = as.integer(cluster)) %>%
    left_join(enframe(celltypes) %>%
                unnest(value) %>%
                rename(celltype = name, cluster = value) %>%
                mutate(cluster = as.integer(cluster))) %>%
    group_by(celltype, id, leiden) %>%
    summarise(score = mean(value)) %>%
    pivot_wider(names_from = celltype, values_from = score) %>%
    ungroup()
  return(df)
}

get_celltype_interaction_scores_norm <- function(mdata, feature_type, celltypes) {
  df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(starts_with('phenocoder_stat_interaction_norm')) %>%
    select(ends_with(feature_type)) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden,
           id = rownames(mdata['phenocoder_combined']$obs)) %>%
    pivot_longer(cols = starts_with('phenocoder_stat_interaction'),
                 names_to = 'type', values_to = 'value') %>%
    separate(col = type,
             into = c('prefix', 'stat', 'interaction', 'norm', 'from', 'to', 'radius', 'feature_type'),
             sep = '_') %>%
    mutate(from = as.integer(from), to = as.integer(to),
           from = ifelse(from %in% celltypes$cancer, 'cancer', 'caf'),
           to = ifelse(to %in% celltypes$cancer, 'cancer', 'caf'),
           interaction_type = ifelse(from == 'cancer' & to == 'cancer', 'cancer-cancer',
                                     ifelse(from == 'caf' & to == 'caf', 'caf-caf', 'cancer-caf'))) %>%
    group_by(interaction_type, leiden, id, radius) %>%
    summarise(score = mean(value)) %>%
    mutate(radius = as.integer(radius))
  return(df)
}

get_celltype_interaction_scores <- function(mdata, feature_type, celltypes) {
  df <- as_tibble(mdata['phenocoder_combined']$layers[['raw']]) %>%
    select(starts_with('phenocoder_stat_interaction'), -contains('norm')) %>%
    select(ends_with(feature_type)) %>%
    mutate(leiden = mdata['phenocoder_combined']$obs$leiden, id = rownames(mdata['phenocoder_combined']$obs)) %>%
    pivot_longer(cols = starts_with('phenocoder_stat_interaction'), names_to = 'type', values_to = 'value') %>%
    separate(col = type, into = c('prefix', 'stat', 'interaction', 'from', 'to', 'radius', 'feature_type'), sep = '_') %>%
    mutate(from = as.integer(from), to = as.integer(to),
           from = ifelse(from %in% celltypes$cancer, 'cancer', 'caf'),
           to = ifelse(to %in% celltypes$cancer, 'cancer', 'caf'),
           interaction_type = (ifelse(from == 'cancer' & to == 'cancer', 'cancer-cancer',
                                      ifelse(from == 'caf' & to == 'caf', 'caf-caf', 'cancer-caf')))) %>%
    group_by(interaction_type, leiden, id, radius) %>%
    summarise(score = mean(value)) %>%
    mutate(radius = as.integer(radius))
  return(df)
}

get_scores <- function(mdata, cell_annotations, proliferation_annotation, lamc2_annotation, duct_scores=TRUE) {
  df_celltype_scores_source <- get_celltype_scores(mdata, 'source', cell_annotations[['source']])
  df_celltype_scores_target <- get_celltype_scores(mdata, 'target', cell_annotations[['target']])
  df_proliferation <- get_proliferation_scores(mdata,
                                               proliferation_annotation[['feature_type']],
                                               proliferation_annotation[['cluster']])
  df_lamc2 <- get_lamc2_scores(mdata,lamc2_annotation[['feature_type']],lamc2_annotation[['cluster']])
  if (duct_scores) {
    df_duct_scores_source <- get_duct_scores(mdata, 'source')
    df_duct_scores_target <- get_duct_scores(mdata, 'target')
  } else {
    df_duct_scores_source <- NULL
    df_duct_scores_target <- NULL
  }
  df_size_source <- get_size_scores(mdata, 'source')
  df_size_target <- get_size_scores(mdata, 'target')

  df_interactions_source_abs <- get_celltype_interaction_scores(mdata_org,
                                                                'source',
                                                                cell_annotations[['source']]) %>%
    pivot_wider(names_from = c(radius, interaction_type), values_from = score, names_prefix = 'radius_')

  df_interactions_target_abs <- get_celltype_interaction_scores(mdata_org,
                                                                'target',
                                                                cell_annotations[['target']]) %>%
    pivot_wider(names_from = c(radius, interaction_type), values_from = score, names_prefix = 'radius_')

  df_interactions_source_norm <- get_celltype_interaction_scores_norm(mdata_org,
                                                                      'source',
                                                                      cell_annotations[['source']]) %>%
    pivot_wider(names_from = c(radius, interaction_type), values_from = score, names_prefix = 'radius_')

  df_interactions_target_norm <- get_celltype_interaction_scores_norm(mdata_org,
                                                                      'target',
                                                                      cell_annotations[['target']]) %>%
    pivot_wider(names_from = c(radius, interaction_type), values_from = score, names_prefix = 'radius_')


  # merge all dfs on leiden and id
  source <- list(celltypes = df_celltype_scores_source,
                 interactions_abs = df_interactions_source_abs,
                 interactions_norm = df_interactions_source_norm,
                 lamc2 = df_lamc2,
                 duct_size = df_duct_scores_source,
                 total_size = df_size_source)
  target <- list(celltypes = df_celltype_scores_target,
                 interactions_abs = df_interactions_target_abs,
                 interactions_norm = df_interactions_target_norm,
                 proliferation = df_proliferation,
                 duct_size = df_duct_scores_target,
                 total_size = df_size_target)
  return(list(source = source, target = target))
}

# Define the rescaling and quantile clipping function
rescale_quantile_clip <- function(x, lower_quantile = 0.05, upper_quantile = 0.95) {
  lower_bound <- quantile(x, lower_quantile, na.rm = TRUE)
  upper_bound <- quantile(x, upper_quantile, na.rm = TRUE)
  x <- pmin(pmax(x, lower_bound), upper_bound)  # Clip values
  rescale(x)  # Rescale to [0, 1]
}

plot_celltype_ratios <- function(list_scores, feature_type) {
  p_celltypes_score <- list_scores[[feature_type]]$celltypes %>%
    ggplot(aes(x = caf, y = cancer, col = factor(leiden))) +
    geom_point() +
    ylab('cancer score') +
    xlab('caf score') +
    theme_minimal()
  return(p_celltypes_score)
}

plot_interactions_scores <- function(list_scores, feature_type) {

  p1_norm_interactions <- list_scores[[feature_type]]$interactions_norm %>%
    group_by(interaction_type, leiden, radius) %>%
    summarise(score = mean(score)) %>%
    ggplot(aes(radius, score, col = leiden, group = leiden)) +
    geom_line() +
    geom_point() +
    facet_wrap(~interaction_type) +
    xlab('Kernel size') +
    ylab('Interaction score norm') +
    theme_minimal()

  p2_norm_interactions <- list_scores[[feature_type]]$interactions_norm %>%
    ggplot(aes(y = score, x = leiden, fill = leiden)) +
    scale_fill_brewer(palette = 'Set3') +
    geom_violin() +
    geom_boxplot(outlier.alpha = 0,
                 fill = 'white', width = 0.1, alpha = 0.5) +
    facet_wrap(interaction_type ~ radius, scales = 'free') +
    ylab('Interaction score norm')

  p1_interactions <- list_scores[[feature_type]]$interactions_abs %>%
    group_by(interaction_type, leiden, radius) %>%
    summarise(score = mean(score)) %>%
    ggplot(aes(radius, score, col = leiden, group = leiden)) +
    geom_line() +
    geom_point() +
    facet_wrap(~interaction_type) +
    xlab('Kernel size') +
    ylab('Interaction score abs') +
    theme_minimal()

  p2_interactions <- list_scores[[feature_type]]$interactions_abs %>%
    ggplot(aes(y = score, x = leiden, fill = leiden)) +
    scale_fill_brewer(palette = 'Set3') +
    geom_violin() +
    geom_boxplot(outlier.alpha = 0,
                 fill = 'white', width = 0.1, alpha = 0.5) +
    facet_wrap(interaction_type ~ radius, scales = 'free') +
    ylab('Interaction score abs')

  return(p1_norm_interactions +
           p2_norm_interactions +
           p1_interactions +
           p2_interactions)

}

plot_interaction_heatmap <- function(list_scores, feature_type, df_features, list_df_de=NULL, list_mahal=NULL) {
  if (feature_type == 'merged') {
    feature_types <- c('source', 'target') %>% set_names()
    df_scores <- map(feature_types, function(x) {
      df_scores <- reduce(list_scores[[x]], left_join, by = c('id', 'leiden')) %>%
        left_join(df_features %>% mutate(id = str_c(well_id, plate_id, sep = '_')))
      colnames(df_scores) <- str_replace(colnames(df_scores), '\\.x', '_abs') %>%
        str_replace('\\.y', '_norm')
      return(df_scores)
    }) %>% bind_rows(.id = 'feature_type_merge')
    feature_cols_merged <- df_scores %>% select(caf:density_chull, cell_count:proliferative_frac) %>% colnames()
    df_scores <- df_scores %>% select(-feature_type_merge) %>%
      group_by(id,leiden,condition) %>% summarise_at(all_of(feature_cols_merged), mean, na.rm=TRUE) %>% ungroup()
    feature_cols <- feature_cols_merged
  } else {
    df_scores <- reduce(list_scores[[feature_type]], left_join, by = c('id', 'leiden')) %>%
      left_join(df_features %>% mutate(id = str_c(well_id, plate_id, sep = '_')))
    colnames(df_scores) <- str_replace(colnames(df_scores), '\\.x', '_abs') %>%
      str_replace('\\.y', '_norm')
    feature_cols <- df_scores %>%
    select(caf:density_chull) %>%
    colnames()
  }

  df_scores_agg <- df_scores %>%
    group_by(condition) %>%
    summarise_at(feature_cols, mean) %>%
    mutate_at(feature_cols, rescale_quantile_clip)

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
  rownames(M) <- df_scores_agg$condition
  colnames(M) <- colnames(M) %>% str_replace_all('_', ' ')

  df_annotations <- tibble(condition = rownames(M)) %>%
    left_join(df_features %>%
                select(conc, timepoint, compound, condition) %>%
                distinct(), by = 'condition') %>%
    mutate(timepoint = str_replace(timepoint, 'Day ', '') %>% as.integer(),
           conc = str_replace(conc, ' µM', '') %>% as.integer()) %>%
    select(conc, timepoint, condition, compound)
  if (!is.null(list_df_de)){
    n_de_features <- map(list_df_de, presto::top_markers, n = 10000, padj_max = 0.005)  %>%
      map(function(x){tibble(n_de=nrow(x))}) %>% bind_rows(.id='condition') %>%
      left_join(list_de$df_features %>%
                  select(condition, conc, timepoint, compound) %>%
                  distinct()) %>%
      mutate(timepoint = str_remove(as.character(timepoint), 'day_') %>% as.integer(),
             conc = str_remove(as.character(conc), '_µM') %>% as.integer()) %>%
      select(-condition)
    df_annotations <- df_annotations  %>%
      left_join(n_de_features, by=c('compound','timepoint','conc'), relationship = 'many-to-one') %>%
      mutate(n_de = log10(n_de+1))
  }
  if (!is.null(list_mahal)){
    M_mahal <- list_mahal$mahal_results$distance
    col_DMSO <- colnames(M_mahal) == 'DMSO'
    M_mahal <- M_mahal[,col_DMSO]
    df_annotations <- df_annotations %>% left_join(M_mahal %>% enframe(name='condition', value='distance_dmso')) %>%
      mutate(distance_dmso=log1p(distance_dmso))
  }
  df_annotations <- as.data.frame(df_annotations)
  rownames(df_annotations) <- df_annotations$condition
  ann_colors <- list(
    timepoint = c("#ecf8fb", "#b2e1e2", "#2ba25f") %>% set_names(levels(df_annotations$timepoint)),
    conc = c("#ecf8fb", "#b2cde3", "#8856a7") %>% set_names(levels(df_annotations$conc)),
    n_de = gray.colors(8) %>% rev(),
    distance_dmso = colorRampPalette(brewer.pal(9,"Blues"))(9),
    compound = colors_compound)

  df_annotations$condition <- NULL
  # colnames_features <- colnames(M)
  # colnames_features <- str_replace(colnames_features,'radius ','@')
  # colnames_features <- str_replace(colnames_features,'n chulls', 'Ducts: Count')
  # colnames_features <- str_replace(colnames_features, 'n pts ', 'Ducts: Cell count ')
  # colnames_features <- str_replace(colnames_features, 'volume chull ', 'Ducts: ')


  p <- pheatmap::pheatmap(M %>% t(),
                     scale = 'row',
                     show_colnames = 0,
                     annotation_col = df_annotations,
                     annotation_colors = ann_colors,
                     cellwidth = 10,
                     cellheight = 10,
                     silent = TRUE)
  return(p)
}