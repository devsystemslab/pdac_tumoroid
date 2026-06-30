#!/usr/bin/env Rscript
# =============================================================================
# Publication figure: Multi-patient drug screen (reduced)
# =============================================================================
# Produces:
#   A) PCA embeddings of phenotypic features (cancer, CAF, drug)
#   B) Pairwise Mahalanobis distance heatmap between conditions
#   C) Top-50 features (by eta^2) across conditions, z-scored
#
# Plus diagnostics:
#   - CV_within vs CV_total scatter, colored by eta^2
#   - CV_within vs CV_total scatter, colored by feature type
#   - eta^2 boxplots per feature type and kernel size
# =============================================================================

library(anndata)
library(tidyverse)
library(patchwork)
library(pheatmap)
library(grid)
library(gridExtra)

source("analysis/multiple_patients/utils.R")

# =============================================================================
# 1. DATA LOADING
# =============================================================================

message("Loading data...")

file <- "data/multiple_patients/anndata/mdata_subset.h5mu"
mdata <- read_mdata(file)
setwd("analysis/multiple_patients/plots")

X <- mdata["phenocoder_combined"]$X |> as.matrix()
# remove duplicate columns which can have different column names but identical values
# identical values (e.g. symmetric interaction features like 4_5 and 5_4).
# Deduplicate on column contents, keeping the first-occurring column name.
dup_cols <- duplicated(t(X))
if (any(dup_cols)) {
  X <- X[, !dup_cols, drop = FALSE]
}

X_pca <- mdata["phenocoder_combined"]$obsm["X_pca"][[1]] |> as.matrix()
feature_names <- colnames(X)

meta <- mdata["phenocoder_combined"]$obs |>
  as_tibble() |>
  mutate(condition = str_c(cancer, caf, drug, sep = "_"))

variance_ratio <- mdata["phenocoder_combined"]$uns$pca$variance_ratio |> unlist()
n_pcs <- length(variance_ratio)

# combined df for embeddings
df <- meta |>
  bind_cols(X_pca |> as_tibble() |> set_names(str_c("PC_", seq_len(ncol(X_pca))))) |>
  bind_cols(
    mdata["phenocoder_combined"]$obsm["X_umap"][[1]] |>
      as_tibble() |> set_names(c("UMAP_1", "UMAP_2"))
  )

# color palette (shared across panels)
ann_colors <- list(
  drug = c("DMSO" = "grey80", "Erlotinib" = "#e8a628", "Osimertinib" = "#d45f5f"),
  cancer = c("P382" = "#a8d8ea", "P388" = "#aa96da", "P506" = "#fcbad3", "P585" = "#ffffd2"),
  caf = c("P382" = "#a8d8ea", "P388" = "#aa96da", "P506" = "#fcbad3", "P585" = "#ffffd2")
)

# =============================================================================
# 2. PCA EMBEDDINGS (Panel A)
# =============================================================================

message("Generating embeddings...")

theme_embed <- theme_void(base_size = 10) +
  theme(
    plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
    legend.position = "bottom",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7)
  )

p_embed_cancer <- ggplot(df, aes(x = PC_1, y = PC_2)) +
  geom_point(aes(fill = cancer), size = 2, shape = 21, color = "black", stroke = 0.3) +
  coord_equal() +
  scale_fill_manual(values = ann_colors$cancer, name = "Cancer") +
  theme_embed +
  ggtitle("Cancer line")

p_embed_caf <- ggplot(df, aes(x = PC_1, y = PC_2)) +
  geom_point(aes(fill = caf), size = 2, shape = 21, color = "black", stroke = 0.3) +
  coord_equal() +
  scale_fill_manual(values = ann_colors$caf, name = "CAF") +
  theme_embed +
  ggtitle("CAF line")

p_embed_drug <- ggplot(df, aes(x = PC_1, y = PC_2)) +
  geom_point(aes(fill = drug), size = 2, shape = 21, color = "black", stroke = 0.3) +
  coord_equal() +
  scale_fill_manual(values = ann_colors$drug, name = "Treatment") +
  theme_embed +
  ggtitle("Treatment")

p_panel_a <- p_embed_cancer + p_embed_caf + p_embed_drug


# =============================================================================
# 3. MAHALANOBIS DISTANCE HEATMAP (Panel B)
# =============================================================================

message("Computing Mahalanobis distances...")

results_mahal <- pairwise.mahalanobis(
  df |> select(starts_with("PC_")) |> as.matrix(),
  df$condition
)

df_annotation <- tibble(condition = rownames(results_mahal$distance)) |>
  separate(condition, into = c("cancer", "caf", "drug"), sep = "_", remove = FALSE)

annotation_col <- df_annotation |>
  select(condition, drug, cancer, caf) |>
  as.data.frame()
rownames(annotation_col) <- annotation_col$condition
annotation_col$condition <- NULL
annotation_col$drug <- factor(annotation_col$drug)
annotation_col$cancer <- factor(annotation_col$cancer)
annotation_col$caf <- factor(annotation_col$caf)

ann_colors_pheatmap <- list(
  drug = ann_colors$drug[levels(annotation_col$drug)],
  cancer = ann_colors$cancer[levels(annotation_col$cancer)],
  caf = ann_colors$caf[levels(annotation_col$caf)]
)

p_panel_b <- pheatmap::pheatmap(
  results_mahal$distance,
  annotation_col = annotation_col,
  annotation_row = annotation_col,
  annotation_colors = ann_colors_pheatmap,
  show_rownames = FALSE,
  show_colnames = FALSE,
  cluster_cols = FALSE,
  cluster_rows = FALSE,
  cellwidth = 8,
  cellheight = 8,
  na_col = "darkred",
  fontsize = 7,
  main = "Pairwise Mahalanobis distance",
  silent = TRUE
)

# =============================================================================
# 4. FEATURE IMPORTANCE — Cohen's d across drug-vs-DMSO contrasts
# =============================================================================
# For each (cancer, caf) background, compute Cohen's d between each drug
# (Erlotinib, Osimertinib) and DMSO. Rank features by mean |d| across all
# contrasts, take the top 50 for Panel C.
#
# eta2_one() is defined here as well — it is used downstream in the
# diagnostic section (Section 8) for variance-explained-by-plate_id.
# =============================================================================

# helper kept here so it's available for the diagnostics section below
eta2_one <- function(vals, grp) {
  keep <- is.finite(vals)
  vals <- vals[keep]
  grp <- grp[keep]

  n <- length(vals)
  if (n < 3) return(NA_real_)
  g_fact <- factor(grp)
  k <- nlevels(g_fact)
  if (k < 2) return(NA_real_)
  if (sd(vals) == 0 || !is.finite(sd(vals))) return(NA_real_)

  grand <- mean(vals)
  sst <- sum((vals - grand)^2)
  g_int <- as.integer(g_fact)
  n_g <- tabulate(g_int, nbins = k)
  sums_g <- as.numeric(rowsum(vals, g_int))
  means_g <- sums_g / n_g
  ssb <- sum(n_g * (means_g - grand)^2)
  ssb / sst
}

message("Computing Cohen's d across drug-vs-DMSO contrasts...")

# enumerate drug-vs-DMSO contrasts (one per cancer x caf background)
meta_conditions <- tibble(condition = unique(meta$condition)) |>
  separate(condition, into = c("cancer", "caf", "drug"), sep = "_", remove = FALSE)

drug_contrasts <- meta_conditions |>
  filter(drug != "DMSO") |>
  inner_join(
    meta_conditions |> filter(drug == "DMSO") |> rename(baseline = condition),
    by = c("cancer", "caf"), suffix = c("_drug", "_dmso")
  ) |>
  mutate(contrast_label = str_c(cancer, "_", caf, ": ", drug_drug, " vs DMSO"))

# per-contrast Cohen's d for every feature
run_cohens_d_contrast <- function(cond_drug, cond_baseline) {
  idx_drug <- which(meta$condition == cond_drug)
  idx_base <- which(meta$condition == cond_baseline)
  if (length(idx_drug) < 2 || length(idx_base) < 2) {
    return(tibble(feature = feature_names, cohens_d = NA_real_))
  }

  X_drug <- X[idx_drug, , drop = FALSE]
  X_base <- X[idx_base, , drop = FALSE]
  n_drug <- nrow(X_drug)
  n_base <- nrow(X_base)
  mean_drug <- colMeans(X_drug)
  mean_base <- colMeans(X_base)
  sd_drug <- apply(X_drug, 2, sd)
  sd_base <- apply(X_base, 2, sd)

  sd_pooled <- sqrt(((n_drug - 1) * sd_drug^2 + (n_base - 1) * sd_base^2) /
                      (n_drug + n_base - 2))
  sd_pooled[sd_pooled == 0] <- NA_real_

  tibble(
    feature = feature_names,
    cohens_d = (mean_drug - mean_base) / sd_pooled
  )
}

df_de_all <- drug_contrasts |>
  mutate(de = map2(condition, baseline, run_cohens_d_contrast)) |>
  unnest(de)

# rank features by mean |Cohen's d| across contrasts
df_feature_rank <- df_de_all |>
  group_by(feature) |>
  summarise(mean_abs_d = mean(abs(cohens_d), na.rm = TRUE), .groups = "drop") |>
  arrange(desc(mean_abs_d))

top_features <- df_feature_rank |>
  slice_head(n = 50) |>
  pull(feature)

# =============================================================================
# 5. TOP-50 FEATURES × CONDITIONS HEATMAP (Panel C)
# =============================================================================

message("Building top-50 features heatmap...")

X_scaled <- scale(X[, top_features])
colnames(X_scaled) <- str_remove(colnames(X_scaled), "phenocoder_") %>% str_remove("stat_")
colnames(X_scaled) <- str_replace(colnames(X_scaled), "^mean","connectivity_mean") %>% str_replace("^std","connectivity_std") %>% str_replace_all("_","-")
df_condition_means <- tibble(condition = meta$condition) |>
  bind_cols(as_tibble(X_scaled)) |>
  group_by(condition) |>
  summarise(across(everything(), mean), .groups = "drop")

mat_condition <- df_condition_means |>
  column_to_rownames("condition") |>
  as.matrix()

ann_condition <- tibble(condition = rownames(mat_condition)) |>
  separate(condition, into = c("cancer", "caf", "drug"), sep = "_", remove = FALSE) |>
  column_to_rownames("condition") |>
  mutate(across(everything(), as.factor)) |>
  as.data.frame()

p_panel_c <- pheatmap::pheatmap(
  t(mat_condition),
  show_rownames = TRUE,
  show_colnames = FALSE,
  cluster_rows = TRUE,
  cluster_cols = FALSE,
  annotation_col = ann_condition,
  annotation_colors = ann_colors_pheatmap,
  main = "Top 50 features (by mean |Cohen's d| across drug contrasts) across conditions (z-scored)",
  fontsize_row = 5,
  fontsize_col = 6,
  fontsize = 7,
  cellwidth = 6,
  cellheight = 6,
  silent = TRUE
)

# =============================================================================
# 6. SAVE INDIVIDUAL PANELS
# =============================================================================

message("Saving individual panels...")

ggsave("fig_panel_a_embeddings.pdf",
  plot = p_panel_a,
  width = 180, height = 60, units = "mm"
)

ggsave("fig_panel_b_mahalanobis.pdf",
  plot = p_panel_b[[4]],
  width = 180, height = 160, units = "mm", limitsize = FALSE
)

ggsave("fig_panel_c_top_features.pdf",
  plot = p_panel_c[[4]],
  width = 180,
  height = max(120, length(top_features) * 3.2),
  units = "mm", limitsize = FALSE
)

# =============================================================================
# 7. COMPOSITE FIGURE
# =============================================================================

message("Assembling composite figure...")

grob_b <- wrap_elements(full = p_panel_b[[4]])
grob_c <- wrap_elements(full = p_panel_c[[4]])

composite <- (
  (p_panel_a + plot_layout(nrow = 1)) /
    (grob_b + grob_c + plot_layout(widths = c(1, 1))) +
    plot_layout(heights = c(1, 3))
) +
  plot_annotation(
    tag_levels = "A",
    theme = theme(plot.tag = element_text(size = 14, face = "bold"))
  )

ggsave("figure_multipatient.pdf",
  plot = composite,
  width = 360, height = 400, units = "mm", limitsize = FALSE
)

# =============================================================================
# 8. FEATURE CV SCATTER + eta^2 DIAGNOSTICS
# =============================================================================
# For each feature, compare:
#   CV_total  = SD / mean across ALL organoids (ignores timepoints)
#   CV_within = mean of per-timepoint CVs (within-group variability)
#
# Features above the diagonal vary mostly between timepoints (informative).
# Features near the diagonal vary mostly within a timepoint (uninformative).
#
# eta^2 from one-way ANOVA (feature ~ plate_id) quantifies the fraction
# of variance explained by timepoint.
# =============================================================================

message("Computing per-feature CV scatter + eta^2 diagnostics...")

# raw feature columns from msg_imputed_combined
feature_cols <- colnames(
  mdata["phenocoder_combined"]$layers["raw"] |> as_tibble()
)

df_feat <- mdata["phenocoder_combined"]$obs |>
  as_tibble() |>
  bind_cols(mdata["phenocoder_combined"]$layers["raw"] |> as_tibble())

# CV_total: across all organoids
cv_total <- df_feat |>
  summarise(across(
    all_of(feature_cols),
    ~ sd(., na.rm = TRUE) / abs(mean(., na.rm = TRUE))
  )) |>
  pivot_longer(everything(), names_to = "feature", values_to = "cv_total")

# CV_within: mean of per-timepoint CVs
cv_within <- df_feat |>
  group_by(plate_id) |>
  summarise(
    across(
      all_of(feature_cols),
      ~ sd(., na.rm = TRUE) / abs(mean(., na.rm = TRUE))
    ),
    .groups = "drop"
  ) |>
  select(-plate_id) |>
  summarise(across(everything(), ~ mean(., na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "cv_within")

# eta^2 per feature, grouped by plate_id (timepoint)
message("Computing eta^2 per feature (grouped by plate_id)...")

df_eta2_plate <- tibble(
  feature = feature_cols,
  eta2 = vapply(
    feature_cols,
    function(f) eta2_one(df_feat[[f]], df_feat$plate_id),
    numeric(1)
  )
)

df_cv_feat <- df_eta2_plate |>
  left_join(cv_total, by = "feature") |>
  left_join(cv_within, by = "feature") |>
  mutate(
    feature_type = case_when(
      str_detect(feature, "interaction") ~ "interaction",
      str_detect(feature, "nhood_z") ~ "neighborhood enrichment",
      str_detect(
        feature,
        "degree|closeness|centrality|stat_mean|stat_std"
      ) ~ "node connectivity",
      str_detect(feature, "moranI") ~ "spatial autocorrelation",
      str_detect(
        feature,
        "^phenocoder_msg_neighbors_imputed_\\d+"
      ) ~ "cell and cluster counts/proportions",
      str_detect(
        feature,
        "^phenocoder_msg_nuclei_imputed_\\d+"
      ) ~ "cell and cluster counts/proportions",
      str_detect(feature, "chull|n_pts|distance_center") ~ "convex hull stats",
      str_detect(feature, "cell_count") ~ "cell and cluster counts/proportions",
      TRUE ~ "other"
    ),
    kernel_size = as.integer(
      str_extract(feature, "(?<=_)(25|50|100|150)$")
    ),
    scope = if_else(is.na(kernel_size), "whole organoid", "local"),
    kernel_size_f = factor(
      kernel_size,
      levels = c(25, 50, 100, 150),
      exclude = NULL
    ) |>
      fct_na_value_to_level("whole organoid")
  )

# --- Plot A: CV scatter colored by eta^2 ------------------------------------
p_cv_eta2 <- ggplot(df_cv_feat, aes(x = cv_within, y = cv_total)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(color = eta2), size = 1.5, alpha = 0.8) +
  scale_color_viridis_c(
    option = "viridis",
    name = expression(eta^2),
    limits = c(0, NA)
  ) +
  labs(
    x = "CV within timepoints (mean per-timepoint CV)",
    y = "CV total (across all organoids)"
  ) +
  scale_x_log10() +
  scale_y_log10() +
  theme_bw(base_size = 9) +
  theme(legend.position = "right")

# --- Plot B: CV scatter colored by feature type -----------------------------
p_cv_type <- ggplot(df_cv_feat, aes(x = cv_within, y = cv_total)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(color = feature_type), size = 1.5, alpha = 0.8) +
  scale_color_brewer(palette = "Set1", name = "Feature type") +
  labs(
    x = "CV within timepoints (mean per-timepoint CV)",
    y = "CV total (across all organoids)"
  ) +
  scale_x_log10() +
  scale_y_log10() +
  theme_bw(base_size = 9) +
  theme(legend.position = "right")

# --- Plot C: eta^2 boxplots per feature type and kernel size ----------------
p_eta2_box <- ggplot(
  df_cv_feat,
  aes(x = feature_type, y = eta2, fill = kernel_size_f)
) +
  geom_boxplot(
    position = position_dodge(preserve = "single"),
    outlier.size = 0.5
  ) +
  scale_fill_brewer(palette = "Set2", name = "Kernel size") +
  labs(
    x = "Feature type",
    y = expression(eta^2),
    title = expression(eta^2 * " by feature type and kernel size")
  ) +
  theme_bw(base_size = 9) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p_cv_es_combined <- (p_cv_eta2 | p_cv_type) / p_eta2_box

ggsave(
  "supp_qc_cv_feature_scatter_eta2.pdf",
  plot = p_cv_es_combined,
  width = 220, height = 220, units = "mm"
)

ggsave(
  "supp_qc_cv_feature_scatter_eta2.png",
  plot = p_cv_es_combined,
  width = 220, height = 220, units = "mm"
)

# =============================================================================
# DONE
# =============================================================================
