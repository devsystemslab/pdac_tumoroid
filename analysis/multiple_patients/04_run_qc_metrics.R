library(anndata)
library(tidyverse)
library(patchwork)
library(ggrepel)

source(
  "analysis/multiple_patients/utils.R"
)
setwd(
  "analysis/multiple_patients/plots"
)
# =============================================================================
# 1. DATA LOADING
# =============================================================================

message("Loading data...")

file <- "data/multiple_patients/anndata/mdata_subset.h5mu"
mdata <- read_mdata(file)

df <- mdata["phenocoder_combined"]$obs |>
  as_tibble() |>
  mutate(condition = str_c(cancer, caf, drug, sep = "_")) |>
  bind_cols(mdata["phenocoder_combined"]$layers["raw"] |> as_tibble())

# color palette
ann_colors <- list(
  drug = c(
    "DMSO" = "grey80",
    "Erlotinib" = "#e8a628",
    "Osimertinib" = "#d45f5f"
  ),
  cancer = c(
    "P382" = "#a8d8ea",
    "P388" = "#aa96da",
    "P506" = "#fcbad3",
    "P585" = "#ffffd2"
  ),
  caf = c(
    "P382" = "#a8d8ea",
    "P388" = "#aa96da",
    "P506" = "#fcbad3",
    "P585" = "#ffffd2"
  )
)

# =============================================================================
# 2. IDENTIFY CLUSTER PROPORTION COLUMNS
# =============================================================================

all_cols <- colnames(df)
cluster_cols <- all_cols[str_detect(all_cols, "^phenocoder_\\d+$")]
cluster_cols_msg <- all_cols[str_detect(all_cols, "^phenocoder_msg_\\d+$")]

message(
  "Found cluster proportion columns: ",
  paste(cluster_cols, collapse = ", ")
)
message("Found msg cluster columns: ", paste(cluster_cols_msg, collapse = ", "))

# =============================================================================
# 3. COMPUTE DERIVED METRICS
# =============================================================================

df <- df |>
  mutate(
    density = cell_count / phenocoder_stat_volume_chull
  )

# =============================================================================
# 4. SAMPLE SIZES
# =============================================================================

message("Computing sample sizes...")

df_n <- df |>
  count(cancer, caf, drug, condition) |>
  arrange(cancer, caf, drug)

p_n <- ggplot(
  df_n,
  aes(x = interaction(caf, drug, sep = "\n"), y = n, fill = drug)
) +
  geom_col(color = "black", linewidth = 0.2) +
  geom_text(aes(label = n), vjust = -0.3, size = 2.5) +
  facet_wrap(~cancer, scales = "free_x", nrow = 1) +
  scale_fill_manual(values = ann_colors$drug, name = "Treatment") +
  labs(x = NULL, y = "n (organoids)", title = "Sample size per condition") +
  theme_bw(base_size = 9) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(
  "supp_qc_sample_sizes.pdf",
  plot = p_n,
  width = 250,
  height = 100,
  units = "mm"
)

# =============================================================================
# 5. BASIC METRICS BOXPLOTS
# =============================================================================

message("Plotting basic metrics...")

df_metrics <- df |>
  select(
    cancer,
    caf,
    drug,
    condition,
    cell_count,
    phenocoder_stat_volume_chull,
    density
  ) |>
  rename(volume = phenocoder_stat_volume_chull) |>
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
  )

p_metrics <- ggplot(
  df_metrics,
  aes(x = interaction(caf, drug, sep = "\n"), y = value, fill = drug)
) +
  geom_boxplot(outlier.size = 0.5, linewidth = 0.3) +
  facet_grid(metric ~ cancer, scales = "free") +
  scale_fill_manual(values = ann_colors$drug, name = "Treatment") +
  labs(x = NULL, y = NULL, title = "Basic organoid metrics across conditions") +
  theme_bw(base_size = 9) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
    strip.text = element_text(face = "bold", size = 8),
    strip.text.y = element_text(angle = 0),
    legend.position = "bottom"
  )

ggsave(
  "supp_qc_basic_metrics.pdf",
  plot = p_metrics,
  width = 280,
  height = 200,
  units = "mm"
)

# =============================================================================
# 6. CLUSTER PROPORTIONS
# =============================================================================

message("Plotting  absolute cluster proportions...")

if (length(cluster_cols) > 0) {
  df_clusters <- df |>
    select(cancer, caf, drug, condition, cell_count, all_of(cluster_cols)) |>
    group_by(cancer, caf, drug, condition) |>
    mutate(across(all_of(cluster_cols), ~ . / cell_count)) |>
    summarise(across(all_of(cluster_cols), mean), .groups = "drop") |>
    pivot_longer(
      cols = all_of(cluster_cols),
      names_to = "cluster",
      values_to = "proportion"
    ) |>
    mutate(cluster = str_replace(cluster, "phenocoder_", "cluster "))

  p_clusters <- ggplot(
    df_clusters,
    aes(x = interaction(caf, drug, sep = "\n"), y = proportion, fill = cluster)
  ) +
    geom_col(position = "stack", color = "black", linewidth = 0.1) +
    facet_wrap(~cancer, scales = "free_x", nrow = 1) +
    scale_fill_brewer(palette = "Set2", name = "phenocoder\ncluster") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(
      x = NULL,
      y = "mean proportion",
      title = "cvae cluster composition across conditions"
    ) +
    theme_bw(base_size = 9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
      strip.text = element_text(face = "bold"),
      legend.position = "right"
    )

  ggsave(
    "supp_qc_cluster_proportions.pdf",
    plot = p_clusters,
    width = 280,
    height = 100,
    units = "mm"
  )
} else {
  message("  no raw cluster proportion columns found, skipping.")
}
message("Plotting message passed cluster proportions...")

if (length(cluster_cols_msg) > 0) {
  df_clusters <- df |>
    select(
      cancer,
      caf,
      drug,
      condition,
      cell_count,
      all_of(cluster_cols_msg)
    ) |>
    group_by(cancer, caf, drug, condition) |>
    mutate(across(all_of(cluster_cols_msg), ~ . / cell_count)) |>
    summarise(across(all_of(cluster_cols_msg), mean), .groups = "drop") |>
    pivot_longer(
      cols = all_of(cluster_cols_msg),
      names_to = "cluster",
      values_to = "proportion"
    ) |>
    mutate(cluster = str_replace(cluster, "phenocoder_", "cluster "))

  p_clusters <- ggplot(
    df_clusters,
    aes(x = interaction(caf, drug, sep = "\n"), y = proportion, fill = cluster)
  ) +
    geom_col(position = "stack", color = "black", linewidth = 0.1) +
    facet_wrap(~cancer, scales = "free_x", nrow = 1) +
    scale_fill_brewer(palette = "Set2", name = "phenocoder\ncluster") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(
      x = NULL,
      y = "mean proportion",
      title = "cvae cluster composition across conditions"
    ) +
    theme_bw(base_size = 9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
      strip.text = element_text(face = "bold"),
      legend.position = "right"
    )

  ggsave(
    "supp_qc_msg_cluster_proportions.pdf",
    plot = p_clusters,
    width = 280,
    height = 100,
    units = "mm"
  )
} else {
  message("  no raw cluster proportion columns found, skipping.")
}
# =============================================================================
# 7. coefficient of variation
# =============================================================================

message("computing coefficient of variation...")

df_cv <- df |>
  group_by(cancer, caf, drug, condition) |>
  summarise(
    n = n(),
    cv_cell_count = sd(cell_count, na.rm = TRUE) /
      mean(cell_count, na.rm = TRUE),
    cv_volume = sd(phenocoder_stat_volume_chull, na.rm = TRUE) /
      mean(phenocoder_stat_volume_chull, na.rm = TRUE),
    cv_density = sd(density, na.rm = TRUE) / mean(density, na.rm = TRUE),
    mean_cell_count = mean(cell_count, na.rm = TRUE),
    sd_cell_count = sd(cell_count, na.rm = TRUE),
    mean_volume = mean(phenocoder_stat_volume_chull, na.rm = TRUE),
    sd_volume = sd(phenocoder_stat_volume_chull, na.rm = TRUE),
    mean_density = mean(density, na.rm = TRUE),
    sd_density = sd(density, na.rm = TRUE),
    .groups = "drop"
  )

df_cv_long <- df_cv |>
  select(cancer, caf, drug, condition, starts_with("cv_")) |>
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
  aes(x = interaction(caf, drug, sep = "\n"), y = cv, fill = drug)
) +
  geom_col(color = "black", linewidth = 0.2) +
  facet_grid(metric ~ cancer, scales = "free_x") +
  scale_fill_manual(values = ann_colors$drug, name = "treatment") +
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    color = "grey40",
    linewidth = 0.3
  ) +
  labs(
    x = NULL,
    y = "cv (sd / mean)",
    title = "within-condition coefficient of variation"
  ) +
  theme_bw(base_size = 9) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
    strip.text = element_text(face = "bold", size = 8),
    strip.text.y = element_text(angle = 0),
    legend.position = "bottom"
  )

ggsave(
  "supp_qc_cv_summary.pdf",
  plot = p_cv,
  width = 280,
  height = 180,
  units = "mm"
)

# =============================================================================
# 8. COEFFICIENT OF VARIATION DISTRIBUTION
# =============================================================================

message("Plotting coefficient of variation distribution...")

df_cv_dist <- df_cv_long |>
  group_by(metric) |>
  summarise(
    mean_cv = mean(cv, na.rm = TRUE),
    sd_cv = sd(cv, na.rm = TRUE),
    .groups = "drop"
  )

p_cv_dist <- ggplot(
  df_cv_long,
  aes(x = cv, fill = metric)
) +
  geom_histogram(bins = 15, color = "black", linewidth = 0.2, alpha = 0.7) +
  geom_vline(
    data = df_cv_dist,
    aes(xintercept = mean_cv, color = metric),
    linetype = "dashed",
    linewidth = 0.5
  ) +
  facet_wrap(~metric, scales = "free") +
  scale_fill_brewer(palette = "Set2", name = "metric") +
  scale_color_brewer(palette = "Set2", name = "metric") +
  geom_vline(
    xintercept = 0.5,
    linetype = "dotted",
    color = "grey40",
    linewidth = 0.3
  ) +
  labs(
    x = "coefficient of variation",
    y = "frequency",
    title = "Distribution of within-condition CV across all conditions"
  ) +
  theme_bw(base_size = 9) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold", size = 8)
  )

ggsave(
  "supp_qc_cv_distribution.pdf",
  plot = p_cv_dist,
  width = 200,
  height = 150,
  units = "mm"
)

message("Mean CV per metric:")
print(df_cv_dist)

# =============================================================================
# 9. CLUSTER PROPORTION STATISTICAL TESTS (Wilcoxon)
# =============================================================================
# For each cancer x CAF combination, test whether drug treatment shifts
# individual cluster proportions relative to DMSO using Wilcoxon rank-sum
# at the organoid level (the correct independent unit).
# =============================================================================

message("Testing cluster proportion shifts...")

if (length(cluster_cols) > 0) {
  # --- Per-organoid cluster proportions ---
  df_props <- df |>
    mutate(across(all_of(cluster_cols), ~ . / cell_count)) |>
    select(cancer, caf, drug, condition, all_of(cluster_cols))

  drugs <- setdiff(unique(df$drug), "DMSO")

  # --- Per-cluster Wilcoxon rank-sum on proportions ---
  wilcox_cluster_results <- expand_grid(
    cancer_line = unique(df$cancer),
    caf_line = unique(df$caf),
    drug_name = drugs,
    cluster = cluster_cols
  ) |>
    pmap_dfr(function(cancer_line, caf_line, drug_name, cluster) {
      props_dmso <- df_props |>
        filter(cancer == cancer_line, caf == caf_line, drug == "DMSO") |>
        pull(!!sym(cluster))
      props_drug <- df_props |>
        filter(cancer == cancer_line, caf == caf_line, drug == drug_name) |>
        pull(!!sym(cluster))

      if (length(props_dmso) < 2 || length(props_drug) < 2) {
        return(tibble(
          cancer = cancer_line,
          caf = caf_line,
          drug = drug_name,
          cluster = cluster,
          mean_dmso = NA_real_,
          mean_drug = NA_real_,
          diff = NA_real_,
          p_value = NA_real_
        ))
      }

      test <- tryCatch(
        wilcox.test(props_drug, props_dmso, exact = FALSE),
        error = function(e) list(p.value = NA_real_)
      )

      tibble(
        cancer = cancer_line,
        caf = caf_line,
        drug = drug_name,
        cluster = cluster,
        mean_dmso = mean(props_dmso, na.rm = TRUE),
        mean_drug = mean(props_drug, na.rm = TRUE),
        diff = mean(props_drug, na.rm = TRUE) - mean(props_dmso, na.rm = TRUE),
        p_value = test$p.value
      )
    })

  wilcox_cluster_results <- wilcox_cluster_results |>
    mutate(
      p_adj = p.adjust(p_value, method = "BH"),
      cluster_label = str_replace(cluster, "phenocoder_", "cluster "),
      contrast = str_c(cancer, "_", caf, ": ", drug, " vs DMSO"),
      sig = case_when(
        p_adj < 0.001 ~ "***",
        p_adj < 0.01 ~ "**",
        p_adj < 0.05 ~ "*",
        TRUE ~ "ns"
      )
    )

  message("Significant per-cluster shifts (p.adj < 0.05):")
  wilcox_cluster_results |>
    filter(p_adj < 0.05) |>
    select(contrast, cluster_label, diff, p_adj, sig) |>
    print(n = 50)

  # --- Plot: Per-cluster dot heatmap ---
  p_wilcox_clusters <- ggplot(
    wilcox_cluster_results,
    aes(x = contrast, y = cluster_label)
  ) +
    geom_point(aes(size = abs(diff), color = diff)) +
    geom_text(aes(label = sig), vjust = -0.8, size = 2) +
    scale_color_gradient2(
      low = "#4a90d9",
      mid = "grey90",
      high = "#d45f5f",
      midpoint = 0,
      name = "Proportion\ndifference"
    ) +
    scale_size_continuous(range = c(0.5, 5), name = "|Difference|") +
    labs(
      x = NULL,
      y = NULL,
      title = "Per-cluster proportion shift (Wilcoxon, drug vs DMSO)"
    ) +
    theme_bw(base_size = 8) +
    theme(
      axis.text.x = element_text(angle = 60, hjust = 1, size = 5),
      axis.text.y = element_text(size = 7),
      legend.position = "right"
    )

  ggsave(
    "supp_qc_cluster_tests.pdf",
    plot = p_wilcox_clusters,
    width = 300,
    height = 150,
    units = "mm",
    limitsize = FALSE
  )

  # --- Save table ---
  write_csv(
    wilcox_cluster_results |>
      select(
        cancer,
        caf,
        drug,
        cluster,
        contrast,
        mean_dmso,
        mean_drug,
        diff,
        p_value,
        p_adj,
        sig
      ),
    "supp_table_cluster_wilcox.csv"
  )

  message("Cluster test outputs saved.")
} else {
  message("  No cluster columns found, skipping statistical tests.")
}

# =============================================================================
# 10. FEATURE CV SCATTER: CV_within vs CV_total + ANOVA
# =============================================================================
# For each feature, compare:
#   CV_total  = SD / mean across ALL organoids (ignores conditions)
#   CV_within = mean of per-condition CVs (the within-group variability)
#
# Features far above the diagonal have variation primarily explained by
# experimental conditions (condition-informative).
# Features near the diagonal vary mostly between organoids within the same
# condition (condition-uninformative).
#
# Additionally, ANOVA from linear model (feature ~ cancer + caf + drug)
# extracts the drug F-statistic specifically, separating drug effect
# from patient identity.
# =============================================================================

message("Computing per-feature CV scatter + ANOVA...")

# get raw feature matrix
feature_cols <- colnames(
  mdata["phenocoder_combined"]$layers["raw"] |> as_tibble()
)

# rebuild df with raw features for CV computation
df_feat <- mdata["phenocoder_combined"]$obs |>
  as_tibble() |>
  mutate(condition = str_c(cancer, caf, drug, sep = "_")) |>
  bind_cols(mdata["phenocoder_combined"]$layers["raw"] |> as_tibble())

# clip 5 - 95 quantiles for all feature cols
df_feat <- df_feat |>
  mutate(across(
    all_of(feature_cols),
    ~ quantile(., probs = c(0.05, 0.95), na.rm = TRUE)
  )) |>
  pivot_longer(
    all_of(feature_cols),
    names_to = "feature",
    values_to = "value"
  ) |>
  group_by(feature) |>
  mutate(
    value = ifelse(value < quantile(value, 0.05), quantile(value, 0.05), value)
  ) |>
  mutate(
    value = ifelse(value > quantile(value, 0.95), quantile(value, 0.95), value)
  ) |>
  ungroup() |>
  pivot_wider(names_from = "feature", values_from = "value")


# CV_total: across all organoids
cv_total <- df_feat |>
  summarise(across(
    all_of(feature_cols),
    ~ sd(., na.rm = TRUE) / abs(mean(., na.rm = TRUE))
  )) |>
  pivot_longer(everything(), names_to = "feature", values_to = "cv_total")

# CV_within: mean of per-condition CVs
cv_within <- df_feat |>
  group_by(condition) |>
  summarise(
    across(
      all_of(feature_cols),
      ~ sd(., na.rm = TRUE) / abs(mean(., na.rm = TRUE))
    ),
    .groups = "drop"
  ) |>
  select(-condition) |>
  summarise(across(everything(), ~ mean(., na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "cv_within")

# --- ANOVA per feature: feature ~ cancer + caf + drug ---
# Extract drug F-statistic to isolate drug effect from patient identity
message("  Running ANOVA (feature ~ cancer + caf + drug) for each feature...")

anova_results <- tibble(
  feature = feature_cols,
  f_stat_drug = NA_real_,
  p_value_drug = NA_real_
)

for (i in seq_along(feature_cols)) {
  feat <- feature_cols[i]
  vals <- df_feat[[feat]]

  # skip if all NA, all identical, or too few values
  n_valid <- sum(!is.na(vals))
  if (n_valid < 3) {
    next
  }
  feat_sd <- sd(vals, na.rm = TRUE)
  if (is.na(feat_sd) || feat_sd == 0) {
    next
  }

  res <- tryCatch(
    {
      fit <- aov(vals ~ df_feat$cancer + df_feat$caf + df_feat$drug)
      s <- summary(fit)[[1]]
      # drug is the 3rd term
      drug_row <- which(rownames(s) == "df_feat$drug")
      if (length(drug_row) == 0) {
        drug_row <- 3
      }
      list(f = s[["F value"]][drug_row], p = s[["Pr(>F)"]][drug_row])
    },
    error = function(e) list(f = NA_real_, p = NA_real_)
  )

  anova_results$f_stat_drug[i] <- res$f
  anova_results$p_value_drug[i] <- res$p
}

anova_results <- anova_results |>
  mutate(p_adj_drug = p.adjust(p_value_drug, method = "BH"))

# --- Build combined dataframe ---
df_cv_feat <- cv_total |>
  left_join(cv_within, by = "feature") |>
  left_join(anova_results, by = "feature") |>
  filter(is.finite(cv_total) & is.finite(cv_within)) |>
  mutate(
    feature_type = case_when(
      str_detect(feature, "interaction") ~ "interaction",
      str_detect(
        feature,
        "centrality|stat_mean_\\d+$|stat_sd\\d+$"
      ) ~ "centrality",
      str_detect(feature, "nhood_z") ~ "neighborhood enrichment",
      str_detect(feature, "degree") ~ "degree",
      str_detect(feature, "stat_z") ~ "spatial autocorrelation",
      str_detect(feature, "^phenocoder_\\d+$") ~ "cluster proportion",
      str_detect(feature, "^phenocoder_msg_\\d+$") ~ "cluster proportion (msg)",
      str_detect(feature, "volume|cell_count|density|chull") ~ "morphological",
      TRUE ~ "other"
    ),
    ratio = cv_total / cv_within,
    neg_log10_p = -log10(pmax(p_adj_drug, 1e-300)),
    anova_sig = case_when(
      is.na(p_adj_drug) ~ "NA",
      p_adj_drug < 0.001 ~ "***",
      p_adj_drug < 0.01 ~ "**",
      p_adj_drug < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

# identify features to label
top_informative <- df_cv_feat |>
  slice_max(ratio, n = 10) |>
  pull(feature)

top_anova <- df_cv_feat |>
  filter(!is.na(f_stat_drug)) |>
  slice_max(f_stat_drug, n = 10) |>
  pull(feature)

top_labels <- unique(c(top_informative, top_anova))

# --- Plot A: colored by feature type ---
p_cv_type <- ggplot(df_cv_feat, aes(x = cv_within, y = cv_total)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(color = feature_type), size = 1.5, alpha = 0.7) +
  geom_text_repel(
    data = df_cv_feat |> filter(feature %in% top_labels),
    aes(label = feature),
    size = 1.8,
    max.overlaps = 20,
    segment.size = 0.2,
    segment.color = "grey60",
    min.segment.length = 0
  ) +
  scale_y_log10() +
  scale_x_log10() +
  scale_color_brewer(palette = "Set1", name = "Feature type") +
  labs(
    x = "CV within conditions (mean per-condition CV)",
    y = "CV total (across all organoids)",
    title = "Feature informativeness by type"
  ) +
  theme_bw(base_size = 9) +
  theme(legend.position = "right")

# --- Plot B: colored by ANOVA drug significance ---
p_cv_anova <- ggplot(df_cv_feat, aes(x = cv_within, y = cv_total)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(aes(color = neg_log10_p), size = 1.5, alpha = 0.7) +
  geom_text_repel(
    data = df_cv_feat |> filter(feature %in% top_labels),
    aes(label = feature),
    size = 1.8,
    max.overlaps = 20,
    segment.size = 0.2,
    segment.color = "grey60",
    min.segment.length = 0
  ) +
  scale_y_log10() +
  scale_x_log10() +
  scale_color_viridis_c(
    option = "inferno",
    name = expression(-log[10](p[adj])),
    direction = -1
  ) +
  labs(
    x = "CV within conditions (mean per-condition CV)",
    y = "CV total (across all organoids)",
    title = "Feature informativeness by drug ANOVA (feature ~ cancer + caf + drug)"
  ) +
  theme_bw(base_size = 9) +
  theme(legend.position = "right")
p_annova_rank <- ggplot(df_cv_feat %>% mutate(rank_p = rank(neg_log10_p))) +
  geom_point(
    aes(x = rank_p, y = neg_log10_p, col = feature_type),
    size = 1.5,
    alpha = 0.7
  ) +
  labs(
    x = "Rank by -log10(p_adj)",
    y = "-log10(p_adj)",
    title = "ANOVA significance rank"
  ) +
  scale_color_brewer(palette = "Set1", name = "Feature type") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50") +
  theme_bw(base_size = 9) +
  theme(legend.position = "right")

p_box_plot_annova <- ggplot(
  df_cv_feat,
  aes(x = feature_type, group = feature_type, y = neg_log10_p)
) +
  geom_boxplot(aes(fill = feature_type)) +
  scale_fill_brewer(palette = "Set1", name = "Feature type") +
  labs(
    x = "Feature type",
    y = "-log10(p_adj)",
    title = "ANOVA significance by feature type"
  ) +
  theme_bw(base_size = 9) +
  theme(
    legend.position = "right",
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

p_cv_combined <- (p_cv_type | p_cv_anova) / (p_annova_rank | p_box_plot_annova)


ggsave(
  "supp_qc_cv_feature_scatter.pdf",
  plot = p_cv_combined,
  width = 220,
  height = 220,
  units = "mm"
)

# save ANOVA table
write_csv(
  df_cv_feat |>
    select(
      feature,
      feature_type,
      cv_within,
      cv_total,
      ratio,
      f_stat_drug,
      p_value_drug,
      p_adj_drug,
      anova_sig
    ) |>
    arrange(desc(f_stat_drug)),
  "supp_table_feature_anova.csv"
)

message("  ANOVA results saved. Top 10 features by drug F-statistic:")
df_cv_feat |>
  filter(!is.na(f_stat_drug)) |>
  slice_max(f_stat_drug, n = 10) |>
  select(
    feature,
    f_stat_drug,
    p_adj_drug,
    anova_sig,
    cv_total,
    cv_within,
    ratio
  ) |>
  print()

# =============================================================================
# 11. SUMMARY TABLE
# =============================================================================

message("saving summary table...")

if (length(cluster_cols) > 0) {
  df_cluster_summary <- df |>
    group_by(cancer, caf, drug, condition) |>
    summarise(across(all_of(cluster_cols), mean), .groups = "drop") |>
    rename_with(~ str_c("mean_prop_", .), all_of(cluster_cols))

  df_cv_full <- df_cv |>
    left_join(df_cluster_summary, by = c("cancer", "caf", "drug", "condition"))
} else {
  df_cv_full <- df_cv
}

df_cv_means <- df_cv |>
  rowwise() |>
  mutate(
    mean_cv_all = mean(c(cv_cell_count, cv_volume, cv_density), na.rm = TRUE)
  ) |>
  ungroup() |>
  select(cancer, caf, drug, condition, mean_cv_all)

df_cv_full <- df_cv_full |>
  left_join(df_cv_means, by = c("cancer", "caf", "drug", "condition"))

write_csv(df_cv_full, "supp_table_qc_summary.csv")
