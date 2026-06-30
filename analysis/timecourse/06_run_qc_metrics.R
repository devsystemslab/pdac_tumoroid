library(anndata)
library(tidyverse)
library(patchwork)
library(ggrepel)

source(
  "analysis/multiple_patients/utils.R"
)
setwd(
  "analysis/timecourse/plots"
)
# =============================================================================
# 1. DATA LOADING
# =============================================================================

message("Loading data...")

file <- "data/timecourse/anndata/mdata_org_chull_mlp_normalized.h5mu"
mdata <- read_mdata(file)

df <- mdata["msg_imputed_combined"]$obs |>
  as_tibble() |>
  bind_cols(mdata["msg_imputed_combined"]$layers["raw"] |> as_tibble())

# get organoids per timepoint that are outlier in volume and cell count
outliers <- df |>
  select(
    well_id,
    plate_id,
    cell_count,
    phenocoder_msg_neighbors_imputed_stat_volume_chull
  ) |>
  rename(volume = phenocoder_msg_neighbors_imputed_stat_volume_chull) |>
  group_by(plate_id) |>
  mutate(
    is_outlier_volume = abs(volume - median(volume)) > 1.5 * IQR(volume),
    is_outlier_cell_count = abs(cell_count - median(cell_count)) >
      1.5 * IQR(cell_count)
  ) |>
  filter(is_outlier_volume | is_outlier_cell_count) |>
  ungroup() |>
  select(well_id, plate_id)

# remove outliers from df
df <- df |>
  anti_join(outliers, by = c("well_id", "plate_id"))

# Load raw features for DPT pseudotime analysis (if available)
df_raw <- NULL
raw_file <- "analysis/data/timecourse_features_raw.csv"
if (file.exists(raw_file)) {
  df_raw <- read_csv(raw_file) |> rename(index = `...1`)
  message("Loaded raw features for DPT pseudotime analysis")
} else {
  message("Raw features file not found at ", raw_file)
  message("DPT pseudotime analysis will be skipped")
}

# =============================================================================
# 5. BASIC METRICS BOXPLOTS
# =============================================================================

message("Plotting basic metrics...")
df <- df |>
  mutate(
    density = cell_count / phenocoder_msg_neighbors_imputed_stat_volume_chull,
    density = ifelse(is.infinite(density) | is.nan(density), NA, density)
  )
df_metrics <- df |>
  select(
    plate_id,
    cell_count,
    phenocoder_msg_neighbors_imputed_stat_volume_chull,
    density
  ) |>
  rename(volume = phenocoder_msg_neighbors_imputed_stat_volume_chull) |>
  pivot_longer(
    cols = c(cell_count, volume, density),
    names_to = "metric",
    values_to = "value"
  ) |>
  mutate(
    metric = factor(
      metric,
      levels = c("cell_count", "volume", "density"),
      labels = c(
        "Cell count",
        "Volume (convex hull)",
        "Density (cells / volume)"
      )
    )
  ) |>
  group_by(metric) |>
  mutate(
    Q1 = quantile(value, 0.25, na.rm = TRUE),
    Q3 = quantile(value, 0.75, na.rm = TRUE),
    IQR = Q3 - Q1,
    lower_bound = Q1 - 0.5 * IQR,
    upper_bound = Q3 + 0.5 * IQR,
    value = ifelse(
      metric == "Density (cells / volume)" &
        (value < lower_bound | value > upper_bound),
      NA,
      value
    )
  ) |>
  ungroup() |>
  select(-Q1, -Q3, -IQR, -lower_bound, -upper_bound)

p_metrics <- ggplot(
  df_metrics,
  aes(x = plate_id, y = value, fill = plate_id)
) +
  geom_boxplot(outlier.size = 0.5, linewidth = 0.3) +
  facet_wrap(~metric, scales = "free_y") +
  labs(x = NULL, y = NULL, title = "Basic organoid metrics across timepoints") +
  theme_bw(base_size = 9) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
    strip.text = element_text(face = "bold", size = 8),
    strip.text.y = element_text(angle = 0),
    legend.position = "bottom"
  )
ggsave(
  "supp_qc_basic_metrics.png",
  plot = p_metrics,
  width = 200,
  height = 100,
  units = "mm"
)

ggsave(
  "supp_qc_basic_metrics.pdf",
  plot = p_metrics,
  width = 280,
  height = 200,
  units = "mm"
)

# =============================================================================
# 7. Coefficient of variation
# =============================================================================

message("Computing coefficient of variation...")

df_cv <- df |>
  group_by(plate_id) |>
  summarise(
    n = n(),
    cv_cell_count = sd(cell_count, na.rm = TRUE) /
      mean(cell_count, na.rm = TRUE),
    cv_volume = sd(
      phenocoder_msg_neighbors_imputed_stat_volume_chull,
      na.rm = TRUE
    ) /
      mean(phenocoder_msg_neighbors_imputed_stat_volume_chull, na.rm = TRUE),
    cv_density = sd(density, na.rm = TRUE) / mean(density, na.rm = TRUE),
    mean_cell_count = mean(cell_count, na.rm = TRUE),
    sd_cell_count = sd(cell_count, na.rm = TRUE),
    mean_volume = mean(
      phenocoder_msg_neighbors_imputed_stat_volume_chull,
      na.rm = TRUE
    ),
    sd_volume = sd(
      phenocoder_msg_neighbors_imputed_stat_volume_chull,
      na.rm = TRUE
    ),
    mean_density = mean(density, na.rm = TRUE),
    sd_density = sd(density, na.rm = TRUE),
    .groups = "drop"
  )

df_cv_long <- df_cv |>
  select(plate_id, starts_with("cv_")) |>
  pivot_longer(
    cols = starts_with("cv_"),
    names_to = "metric",
    values_to = "cv"
  ) |>
  mutate(
    metric = factor(
      metric,
      levels = c("cv_cell_count", "cv_volume", "cv_density"),
      labels = c("cell count", "volume", "density")
    )
  )

p_cv <- ggplot(
  df_cv_long,
  aes(x = plate_id, y = cv, fill = plate_id)
) +
  geom_col(color = "black", linewidth = 0.2) +
  facet_wrap(~metric, scales = "free_y") +
  labs(
    x = NULL,
    y = "cv (sd / mean)",
    title = "Within-timepoint coefficient of variation"
  ) +
  theme_bw(base_size = 9) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
    strip.text = element_text(face = "bold", size = 8),
    legend.position = "bottom"
  )
ggsave(
  "supp_qc_cv_summary.png",
  plot = p_cv,
  width = 200,
  height = 100,
  units = "mm"
)

ggsave(
  "supp_qc_cv_summary.pdf",
  plot = p_cv,
  width = 200,
  height = 100,
  units = "mm"
)

# =============================================================================
# 10. FEATURE CV SCATTER: CV_within vs CV_total + ANOVA
# =============================================================================
# For each feature, compare:
#   CV_total  = SD / mean across ALL organoids (ignores timepoints)
#   CV_within = mean of per-timepoint CVs (the within-group variability)
#
# Features far above the diagonal have variation primarily explained by
# timepoint (condition-informative).
# Features near the diagonal vary mostly between organoids within the same
# timepoint (condition-uninformative).
#
# Additionally, one-way ANOVA (feature ~ plate_id) provides a formal test
# of whether timepoint explains significant variance for each feature.
# =============================================================================

message("Computing per-feature CV scatter + ANOVA...")

# get raw feature columns
feature_cols <- colnames(
  mdata["msg_imputed_combined"]$layers["raw"] |> as_tibble()
)

# rebuild df with raw features for CV computation
df_feat <- mdata["msg_imputed_combined"]$obs |>
  as_tibble() |>
  bind_cols(mdata["msg_imputed_combined"]$layers["raw"] |> as_tibble())

# remove outliers from df_feat
# df_feat <- df_feat |>
#   anti_join(outliers, by = c("well_id", "plate_id"))

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

# =============================================================================
# 10D. EFFECT SIZES: eta^2, omega^2, Cohen's f
# =============================================================================
# With N in the thousands, p-values are dominated by sample size. Effect
# sizes answer the question we actually care about: how much of each
# feature's variance is explained by timepoint?
#
#   eta^2    = SSB / SST                    (proportion variance explained)
#   omega^2  = (SSB - (k-1)*MSW) / (SST + MSW)
#              bias-corrected; small/neg for null effects
#   Cohen's f = sqrt(eta^2 / (1 - eta^2))    (conventional thresholds:
#              0.10 small, 0.25 medium, 0.40 large)
#
# Note: for one-way ANOVA, eta^2 == partial eta^2 (only one factor).
# Cohen's d is pairwise and doesn't generalize cleanly to 8+ groups;
# Cohen's f is the multi-group analog and is reported instead.
# =============================================================================

message("Computing effect sizes (eta^2, omega^2, Cohen's f)...")

effect_size_one <- function(vals, grp) {
  keep <- is.finite(vals)
  vals <- vals[keep]
  grp <- grp[keep]

  n <- length(vals)
  if (n < 3) {
    return(list(eta2 = NA_real_, omega2 = NA_real_, cohen_f = NA_real_))
  }
  g_fact <- factor(grp)
  k <- nlevels(g_fact)
  if (k < 2) {
    return(list(eta2 = NA_real_, omega2 = NA_real_, cohen_f = NA_real_))
  }
  feat_sd <- sd(vals)
  if (is.na(feat_sd) || feat_sd == 0) {
    return(list(eta2 = NA_real_, omega2 = NA_real_, cohen_f = NA_real_))
  }

  grand <- mean(vals)
  sst <- sum((vals - grand)^2)
  g_int <- as.integer(g_fact)
  n_g <- tabulate(g_int, nbins = k)
  sums_g <- as.numeric(rowsum(vals, g_int))
  means_g <- sums_g / n_g
  ssb <- sum(n_g * (means_g - grand)^2)
  ssw <- sst - ssb
  msw <- ssw / (n - k)

  eta2 <- ssb / sst
  omega2 <- (ssb - (k - 1) * msw) / (sst + msw)
  cohen_f <- if (eta2 < 1) sqrt(eta2 / (1 - eta2)) else NA_real_

  list(eta2 = eta2, omega2 = omega2, cohen_f = cohen_f)
}

effect_results <- tibble(
  feature = feature_cols,
  eta2 = NA_real_,
  omega2 = NA_real_,
  cohen_f = NA_real_
)

for (i in seq_along(feature_cols)) {
  feat <- feature_cols[i]
  res <- effect_size_one(df_feat[[feat]], df_feat$plate_id)
  effect_results$eta2[i] <- res$eta2
  effect_results$omega2[i] <- res$omega2
  effect_results$cohen_f[i] <- res$cohen_f
}

df_cv_feat_es <- effect_results %>%
  left_join(cv_total) %>%
  left_join(cv_within) %>%
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
    scope = if_else(is.na(kernel_size), "whole organoid", "local")
  )

df_cv_feat_es <- df_cv_feat_es %>%
  mutate(
    kernel_size_f = factor(
      kernel_size,
      levels = c(25, 50, 100, 150),
      exclude = NULL # keep NA as its own level
    ) %>%
      fct_na_value_to_level("whole organoid")
  )


# --- Plot A: CV scatter colored by omega^2 ----------------------------------

p_cv_omega <- ggplot(df_cv_feat_es, aes(x = cv_within, y = cv_total)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(color = omega2), size = 1.5, alpha = 0.8) +
  scale_color_viridis_c(
    option = "viridis",
    name = expression(omega^2),
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

p_cv_type <- ggplot(df_cv_feat_es, aes(x = cv_within, y = cv_total)) +
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

p_omega_box <- ggplot(
  df_cv_feat_es,
  aes(x = feature_type, y = omega2, fill = kernel_size_f)
) +
  geom_boxplot(
    position = position_dodge(preserve = "single"),
    outlier.size = 0.5
  ) +
  scale_fill_brewer(palette = "Set2", name = "Kernel size") +
  labs(
    x = "Feature type",
    y = expression(omega^2),
    title = expression(omega^2 * " by feature type and kernel size")
  ) +
  theme_bw(base_size = 9) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p_cv_es_combined <- (p_cv_omega | p_cv_type) / p_omega_box

ggsave(
  "supp_qc_cv_feature_scatter_effect_size.pdf",
  plot = p_cv_es_combined,
  width = 220,
  height = 220,
  units = "mm"
)

ggsave(
  "supp_qc_cv_feature_scatter_effect_size.png",
  plot = p_cv_es_combined,
  width = 220,
  height = 220,
  units = "mm"
)
