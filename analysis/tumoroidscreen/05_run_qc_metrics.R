# =============================================================================
# 1. DATA LOADING
# =============================================================================

message("Loading data...")

library(anndata)
library(tidyverse)
library(patchwork)
library(ggrepel)

source("analysis/pilotscreen/utils.R")

dir_screen <- "data/tumoroidscreen"
dir_adata <- str_c(dir_screen, "anndata", sep = "/")
dir_metafiles <- "metafiles"

setwd(str_c(dir_screen, "plots", sep = "/"))

mdata_org <- read_mdata(str_c(dir_adata, "mdata_org_combined.h5mu", sep = "/"))

df_plate_layouts <- tibble(
  plate = c("HM001", "HM002", "HM003", "HM004", "HM005", "HM006"),
  PickSet = c(
    "RyoPlate1",
    "RyoPlate1",
    "RyoPlate2",
    "RyoPlate2",
    "RyoPlate3",
    "RyoPlate3"
  )
)
df_drugs <- readxl::read_excel(
  str_c(dir_metafiles, "RO_lib_layouts_withMeta.xlsx", sep = "/")
) |>
  left_join(df_plate_layouts, relationship = "many-to-many") |>
  mutate(
    well = str_c(LETTERS[plateRow_dest], sprintf("%02d", plateColumn_dest))
  ) |>
  rename(plate_id = plate, well_id = well)

df <- mdata_org["phenocoder_combined"]$obs |>
  as_tibble() |>
  left_join(df_drugs, by = c("plate_id", "well_id")) |>
  mutate(
    compound = ifelse(negative_control == "True", "DMSO", CODENAME),
    plate = plate_id,
    is_dmso = negative_control == "True"
  ) |>
  bind_cols(mdata_org["phenocoder_combined"]$layers["raw"] |> as_tibble())

# =============================================================================
# 2. QC FUNCTION
# =============================================================================

compute_qc_metrics <- function(df, suffix, cycle_label) {
  message(paste(
    "\n========== Processing",
    suffix,
    "—",
    cycle_label,
    "==========\n"
  ))

  all_cols <- colnames(df)
  cell_count_col <- paste0("cell_count_", suffix)
  volume_col <- paste0("phenocoder_stat_volume_chull_", suffix)
  cluster_cols <- all_cols[str_detect(
    all_cols,
    paste0("^phenocoder_\\d+_", suffix, "$")
  )]
  cluster_cols_msg <- all_cols[str_detect(
    all_cols,
    paste0("^phenocoder_msg_\\d+_", suffix, "$")
  )]

  if (!cell_count_col %in% colnames(df)) {
    message(
      "Column ",
      cell_count_col,
      " not found. Skipping QC for suffix: ",
      suffix
    )
    return(NULL)
  }

  df_work <- df |>
    rename(
      cell_count = all_of(cell_count_col),
      volume = all_of(volume_col)
    ) |>
    mutate(density = cell_count / volume)

  suffix_str <- paste0("_", suffix)

  # -------------------------------------------------------------------------
  # 3. SAMPLE SIZES
  # -------------------------------------------------------------------------

  message("  Computing sample sizes...")

  df_n <- df_work |>
    count(plate, is_dmso) |>
    mutate(group = ifelse(is_dmso, "DMSO", "compound"))

  p_n <- ggplot(df_n, aes(x = plate, y = n, fill = group)) +
    geom_col(color = "black", linewidth = 0.2) +
    geom_text(
      aes(label = n),
      position = position_stack(vjust = 0.5),
      size = 2.5
    ) +
    scale_fill_manual(
      values = c("DMSO" = "#d62728", "compound" = "#aec7e8"),
      name = NULL
    ) +
    labs(
      x = NULL,
      y = "n (organoids)",
      title = paste("Sample size per plate —", cycle_label)
    ) +
    theme_bw(base_size = 9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )

  ggsave(
    paste0("supp_qc_sample_sizes", suffix_str, ".pdf"),
    plot = p_n,
    width = 180,
    height = 120,
    units = "mm"
  )

  # -------------------------------------------------------------------------
  # 4. BASIC METRICS — DMSO across plates + Kruskal-Wallis
  # -------------------------------------------------------------------------

  message("  Plotting basic metrics...")

  df_metrics <- df_work |>
    select(plate, is_dmso, cell_count, volume, density) |>
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
      value = ifelse(
        metric == "Density (cells / volume)" & value > Q3 + 0.75 * IQR,
        NA,
        value
      )
    ) |>
    ungroup() |>
    select(-Q1, -Q3, -IQR)

  p_metrics <- ggplot(
    df_metrics,
    aes(x = plate, y = value, fill = is_dmso)
  ) +
    geom_boxplot(outlier.size = 0.5, linewidth = 0.3) +
    facet_wrap(~metric, scales = "free_y") +
    scale_fill_manual(
      values = c("TRUE" = "#d62728", "FALSE" = "#aec7e8"),
      labels = c("TRUE" = "DMSO", "FALSE" = "compound"),
      name = NULL
    ) +
    labs(
      x = NULL,
      y = NULL,
      title = paste("Basic organoid metrics —", cycle_label)
    ) +
    theme_bw(base_size = 9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )

  ggsave(
    paste0("supp_qc_basic_metrics", suffix_str, ".pdf"),
    plot = p_metrics,
    width = 220,
    height = 150,
    units = "mm"
  )

  p_metrics_dmso <- ggplot(
    df_metrics |> filter(is_dmso),
    aes(x = plate, y = value)
  ) +
    geom_boxplot(outlier.alpha = 0) +
    geom_jitter(size = 0.5, alpha = 0.4) +
    facet_wrap(~metric, scales = "free_y") +
    ylim(c(0, NA)) +
    labs(
      x = NULL,
      y = NULL,
      title = paste("DMSO controls across plates —", cycle_label)
    ) +
    theme_bw(base_size = 6) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  ggsave(
    paste0("supp_qc_basic_metrics_dmso", suffix_str, ".pdf"),
    plot = p_metrics_dmso,
    width = 220,
    height = 120,
    units = "mm"
  )

  message("  Kruskal-Wallis test (DMSO across plates):")
  kw_results <- df_metrics |>
    filter(is_dmso) |>
    group_by(metric) |>
    summarise(
      p_value = kruskal.test(value ~ plate)$p.value,
      .groups = "drop"
    ) |>
    mutate(p_adj = p.adjust(p_value, method = "BH"))
  print(kw_results)

  # -------------------------------------------------------------------------
  # 5. CLUSTER PROPORTIONS
  # -------------------------------------------------------------------------

  plot_cluster_props <- function(cols, prefix, file_tag) {
    if (length(cols) == 0) {
      message("  No cluster columns found for ", file_tag, ", skipping.")
      return(invisible(NULL))
    }
    df_clust <- df_work |>
      select(plate, compound, is_dmso, cell_count, all_of(cols)) |>
      group_by(plate, compound) |>
      mutate(across(all_of(cols), ~ . / cell_count)) |>
      summarise(across(all_of(cols), mean), .groups = "drop") |>
      pivot_longer(
        cols = all_of(cols),
        names_to = "cluster",
        values_to = "proportion"
      ) |>
      mutate(
        cluster = str_replace(cluster, fixed(prefix), "cluster "),
        cluster = str_replace(cluster, paste0("_", suffix, "$"), "")
      )

    p <- ggplot(df_clust, aes(x = plate, y = proportion, fill = cluster)) +
      geom_col(position = "stack", color = "black", linewidth = 0.1) +
      facet_wrap(~compound, scales = "free_x") +
      scale_fill_brewer(palette = "Set2", name = "cluster") +
      scale_x_discrete(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0)) +
      labs(
        x = NULL,
        y = "mean proportion",
        title = paste(prefix, "composition —", cycle_label)
      ) +
      theme_bw(base_size = 9) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
        strip.text = element_text(face = "bold"),
        legend.position = "right"
      )

    ggsave(
      paste0(file_tag, suffix_str, ".pdf"),
      plot = p,
      width = 280,
      height = 200,
      units = "mm"
    )
  }

  plot_cluster_props(cluster_cols, "phenocoder_", "supp_qc_cluster_proportions")
  plot_cluster_props(
    cluster_cols_msg,
    "phenocoder_msg_",
    "supp_qc_msg_cluster_proportions"
  )

  # -------------------------------------------------------------------------
  # 6. COEFFICIENT OF VARIATION — within plate (DMSO) vs total
  # -------------------------------------------------------------------------

  message("  Computing coefficient of variation...")

  df_cv <- df_work |>
    group_by(plate, is_dmso) |>
    summarise(
      n = n(),
      cv_cell_count = sd(cell_count, na.rm = TRUE) /
        mean(cell_count, na.rm = TRUE),
      cv_volume = sd(volume, na.rm = TRUE) / mean(volume, na.rm = TRUE),
      cv_density = sd(density, na.rm = TRUE) / mean(density, na.rm = TRUE),
      mean_cell_count = mean(cell_count, na.rm = TRUE),
      sd_cell_count = sd(cell_count, na.rm = TRUE),
      mean_volume = mean(volume, na.rm = TRUE),
      sd_volume = sd(volume, na.rm = TRUE),
      .groups = "drop"
    )

  p_cv <- df_cv |>
    select(plate, is_dmso, starts_with("cv_")) |>
    pivot_longer(starts_with("cv_"), names_to = "metric", values_to = "cv") |>
    mutate(
      metric = factor(
        metric,
        levels = c("cv_cell_count", "cv_volume", "cv_density"),
        labels = c("cell count", "volume", "density")
      )
    ) |>
    ggplot(aes(x = plate, y = cv, fill = is_dmso)) +
    geom_col(position = "dodge", color = "black", linewidth = 0.2) +
    geom_hline(
      yintercept = 0.5,
      linetype = "dashed",
      color = "grey40",
      linewidth = 0.3
    ) +
    facet_wrap(~metric, scales = "free_y") +
    scale_fill_manual(
      values = c("TRUE" = "#d62728", "FALSE" = "#aec7e8"),
      labels = c("TRUE" = "DMSO", "FALSE" = "compound"),
      name = NULL
    ) +
    labs(
      x = NULL,
      y = "CV (sd / mean)",
      title = paste("Within-plate coefficient of variation —", cycle_label)
    ) +
    theme_bw(base_size = 9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom"
    )

  ggsave(
    paste0("supp_qc_cv_summary", suffix_str, ".pdf"),
    plot = p_cv,
    width = 220,
    height = 120,
    units = "mm"
  )

  # -------------------------------------------------------------------------
  # 7. FEATURE CV SCATTER + EFFECT SIZES: batch noise vs treatment signal
  # -------------------------------------------------------------------------

  message(
    "  Computing per-feature CV scatter + effect sizes (batch vs compound)..."
  )

  all_raw_cols <- colnames(
    mdata_org["phenocoder_combined"]$layers["raw"] |> as_tibble()
  )
  feature_cols <- all_raw_cols[str_detect(
    all_raw_cols,
    paste0("_", suffix, "$")
  )]
  feature_cols <- setdiff(feature_cols, c(cell_count_col, volume_col))

  message("  Found ", length(feature_cols), " features for suffix: ", suffix)

  if (length(feature_cols) == 0) {
    message("  No features found, skipping CV scatter.")
    write_csv(df_cv, paste0("supp_table_qc_summary", suffix_str, ".csv"))
    return(NULL)
  }

  # --- CV helper (pooled CV per feature over the supplied organoid set) -----
  .cv_fun <- function(x) sd(x, na.rm = TRUE) / abs(mean(x, na.rm = TRUE))

  # ---- DMSO population -----------------------------------------------------
  cv_total_dmso <- df_work |>
    filter(is_dmso) |>
    summarise(across(all_of(feature_cols), .cv_fun)) |>
    pivot_longer(
      everything(),
      names_to = "feature",
      values_to = "cv_total_dmso"
    )

  cv_within_dmso <- df_work |>
    filter(is_dmso) |>
    group_by(plate) |>
    filter(n() >= 2) |>
    summarise(across(all_of(feature_cols), .cv_fun), .groups = "drop") |>
    select(-plate) |>
    summarise(across(everything(), ~ mean(., na.rm = TRUE))) |>
    pivot_longer(
      everything(),
      names_to = "feature",
      values_to = "cv_within_dmso"
    )

  # ---- Compound population -------------------------------------------------
  cv_total_compound <- df_work |>
    filter(!is_dmso) |>
    summarise(across(all_of(feature_cols), .cv_fun)) |>
    pivot_longer(
      everything(),
      names_to = "feature",
      values_to = "cv_total_compound"
    )

  cv_within_compound <- df_work |>
    filter(!is_dmso) |>
    group_by(compound) |>
    filter(n() >= 2) |>
    summarise(across(all_of(feature_cols), .cv_fun), .groups = "drop") |>
    select(-compound) |>
    summarise(across(everything(), ~ mean(., na.rm = TRUE))) |>
    pivot_longer(
      everything(),
      names_to = "feature",
      values_to = "cv_within_compound"
    )

  # ---- Back-compat: pooled over ALL organoids ------------------------------
  cv_total <- df_work |>
    summarise(across(all_of(feature_cols), .cv_fun)) |>
    pivot_longer(everything(), names_to = "feature", values_to = "cv_total")

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
    if (k < 2 || sd(vals) == 0) {
      return(list(eta2 = NA_real_, omega2 = NA_real_, cohen_f = NA_real_))
    }
    grand <- mean(vals)
    sst <- sum((vals - grand)^2)
    g_int <- as.integer(g_fact)
    n_g <- tabulate(g_int, nbins = k)
    means_g <- as.numeric(rowsum(vals, g_int)) / n_g
    ssb <- sum(n_g * (means_g - grand)^2)
    ssw <- sst - ssb
    msw <- ssw / (n - k)
    eta2 <- ssb / sst
    omega2 <- (ssb - (k - 1) * msw) / (sst + msw)
    cohen_f <- if (eta2 < 1) sqrt(eta2 / (1 - eta2)) else NA_real_
    list(eta2 = eta2, omega2 = omega2, cohen_f = cohen_f)
  }

  message("  Computing eta^2 batch (DMSO across plates) and eta^2 compound...")

  df_work_dmso <- df_work |> filter(is_dmso)

  effect_batch <- tibble(feature = feature_cols, eta2_batch = NA_real_)
  effect_compound <- tibble(feature = feature_cols, eta2_compound = NA_real_)

  for (i in seq_along(feature_cols)) {
    feat <- feature_cols[i]
    effect_batch$eta2_batch[i] <- effect_size_one(
      df_work_dmso[[feat]],
      df_work_dmso$plate
    )$eta2
    effect_compound$eta2_compound[i] <- effect_size_one(
      df_work[[feat]],
      df_work$compound
    )$eta2
  }

  feat_suffix_pattern <- paste0("_", suffix, "$")

  df_cv_feat_es <- cv_total |>
    left_join(cv_within_dmso, by = "feature") |>
    left_join(cv_total_dmso, by = "feature") |>
    left_join(cv_within_compound, by = "feature") |>
    left_join(cv_total_compound, by = "feature") |>
    left_join(effect_batch, by = "feature") |>
    left_join(effect_compound, by = "feature") |>
    mutate(cv_within = cv_within_dmso) |> # back-compat alias
    filter(is.finite(cv_total) & is.finite(cv_within)) |>
    mutate(
      feature_short = str_replace(feature, feat_suffix_pattern, ""),
      feature_type = case_when(
        str_detect(feature, "interaction") ~ "interaction",
        str_detect(
          feature,
          "centrality|degree|stat_mean|stat_std"
        ) ~ "node connectivity",
        str_detect(feature, "nhood_z") ~ "neighborhood enrichment",
        str_detect(feature, "stat_z") ~ "spatial autocorrelation",
        str_detect(
          feature,
          "chull|chulls|n_pts|distance_center"
        ) ~ "convex hull stats",
        str_detect(
          feature,
          "^phenocoder_msg_\\d+"
        ) ~ "cluster proportion (msg)",
        str_detect(feature, "^phenocoder_\\d+") ~ "cluster proportion",
        str_detect(feature, "volume|cell_count|density") ~ "morphological",
        TRUE ~ "other"
      ),
      kernel_size = as.integer(str_extract(
        feature_short,
        "(?<=_)(25|50|100|150)$"
      )),
      kernel_size = ifelse(
        feature_type == "convex hull stats",
        100,
        kernel_size
      ),
      scope = if_else(is.na(kernel_size), "whole organoid", "local"),
      signal_to_noise = eta2_compound / (eta2_batch + 1e-6)
    ) |>
    mutate(
      kernel_size_f = factor(
        kernel_size,
        levels = c(25, 50, 100, 150),
        exclude = NULL
      ) |>
        fct_na_value_to_level("whole organoid")
    )

  # Plot A: DMSO — within-plate CV vs total DMSO CV, colored by eta^2 batch
  p_cv_batch <- ggplot(
    df_cv_feat_es |>
      filter(is.finite(cv_within_dmso), is.finite(cv_total_dmso)) |>
      arrange(eta2_batch),
    aes(x = cv_within_dmso, y = cv_total_dmso)
  ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey50"
    ) +
    geom_point(aes(color = eta2_batch), size = 1, alpha = 0.8) +
    scale_color_viridis_c(
      option = "magma",
      name = expression(eta^2 * " batch"),
      limits = c(0, NA),
      direction = -1
    ) +
    geom_text_repel(
      data = df_cv_feat_es |>
        filter(is.finite(cv_within_dmso), is.finite(cv_total_dmso)) |>
        slice_max(eta2_batch, n = 10),
      aes(label = feature_short),
      size = 1.8,
      max.overlaps = 15,
      segment.size = 0.2,
      segment.color = "grey60"
    ) +
    scale_x_log10() +
    scale_y_log10() +
    coord_fixed() +
    labs(
      x = "CV within plates — DMSO (batch noise)",
      y = "CV total — all DMSO",
      title = paste("Batch variance —", cycle_label)
    ) +
    theme_bw(base_size = 9) +
    theme(
      legend.position = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  # Plot B: Compound — within-compound CV vs total compound CV, colored by eta^2 compound
  p_cv_compound <- ggplot(
    df_cv_feat_es |>
      filter(is.finite(cv_within_compound), is.finite(cv_total_compound)) |>
      arrange(eta2_compound),
    aes(x = cv_within_compound, y = cv_total_compound)
  ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey50"
    ) +
    geom_point(aes(color = eta2_compound), size = 1, alpha = 0.8) +
    scale_color_viridis_c(
      option = "viridis",
      name = expression(eta^2 * " compound"),
      limits = c(0, NA)
    ) +
    geom_text_repel(
      data = df_cv_feat_es |>
        filter(is.finite(cv_within_compound), is.finite(cv_total_compound)) |>
        slice_max(eta2_compound, n = 10),
      aes(label = feature_short),
      size = 1.8,
      max.overlaps = 15,
      segment.size = 0.2,
      segment.color = "grey60"
    ) +
    scale_x_log10() +
    scale_y_log10() +
    coord_fixed() +
    labs(
      x = "CV within compound (replicate noise)",
      y = "CV total — all compounds",
      title = paste("Treatment signal —", cycle_label)
    ) +
    theme_bw(base_size = 9) +
    theme(
      legend.position = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  # Plot C: feature type (descriptive — uses back-compat pooled CVs)
  p_cv_type <- ggplot(
    df_cv_feat_es |> arrange(desc(feature_type)),
    aes(x = cv_within, y = cv_total)
  ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey50"
    ) +
    geom_point(aes(color = feature_type), size = 1, alpha = 0.8) +
    scale_color_brewer(palette = "Set1", name = "Feature type") +
    scale_x_log10() +
    scale_y_log10() +
    coord_fixed() +
    labs(
      x = "CV within plates — DMSO (batch noise)",
      y = "CV total (all organoids)"
    ) +
    theme_bw(base_size = 9) +
    theme(
      legend.position = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  # Plot D: signal vs noise per feature (key reviewer plot)
  p_signal_noise <- ggplot(
    df_cv_feat_es,
    aes(x = eta2_batch, y = eta2_compound, color = feature_type)
  ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey50"
    ) +
    geom_point(size = 1, alpha = 0.8) +
    geom_text_repel(
      data = df_cv_feat_es |> slice_max(signal_to_noise, n = 10),
      aes(label = feature_short),
      size = 1.8,
      max.overlaps = 15,
      segment.size = 0.2,
      segment.color = "grey60"
    ) +
    scale_color_brewer(palette = "Set1", name = "Feature type") +
    labs(
      x = expression(eta^2 * " batch (DMSO across plates)"),
      y = expression(eta^2 * " compound (treatment)"),
      title = "Treatment signal vs batch noise — points above diagonal are drug-responsive"
    ) +
    theme_bw(base_size = 9) +
    theme(
      legend.position = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  p_cv_combined <- (p_cv_batch | p_cv_compound) / (p_cv_type | p_signal_noise)

  ggsave(
    paste0("supp_qc_cv_feature_scatter", suffix_str, ".pdf"),
    plot = p_cv_combined,
    width = 220,
    height = 220,
    units = "mm"
  )

  # -------------------------------------------------------------------------
  # 7b. EFFECT SIZE DISTRIBUTION: per compound vs DMSO null (eta²)
  # -------------------------------------------------------------------------
  # For each compound: eta² per feature (compound wells vs all DMSO wells)
  # DMSO null: per-plate leave-one-out eta² (one DMSO plate vs remaining)
  # -------------------------------------------------------------------------

  message("  Computing per-compound eta² distributions vs DMSO null...")

  compute_eta2_two_groups <- function(mat_a, mat_b) {
    n_a <- nrow(mat_a)
    n_b <- nrow(mat_b)
    mean_a <- colMeans(mat_a, na.rm = TRUE)
    mean_b <- colMeans(mat_b, na.rm = TRUE)
    grand <- (n_a * mean_a + n_b * mean_b) / (n_a + n_b)
    ssb <- n_a * (mean_a - grand)^2 + n_b * (mean_b - grand)^2
    sst <- colSums(sweep(mat_a, 2, grand)^2, na.rm = TRUE) +
      colSums(sweep(mat_b, 2, grand)^2, na.rm = TRUE)
    ssb / (sst + 1e-10)
  }

  dmso_mat <- df_work |>
    filter(is_dmso) |>
    select(all_of(feature_cols)) |>
    as.matrix()
  compounds_vec <- unique(df_work$compound[!df_work$is_dmso])

  df_eta2_cmpd <- map_dfr(compounds_vec, function(cmpd) {
    mat_c <- df_work |>
      filter(compound == cmpd) |>
      select(all_of(feature_cols)) |>
      as.matrix()
    if (nrow(mat_c) == 0) {
      return(tibble())
    }
    tibble(
      compound = cmpd,
      feature = feature_cols,
      eta2 = compute_eta2_two_groups(mat_c, dmso_mat),
      group = "compound"
    )
  })

  dmso_plates_vec <- unique(df_work$plate[df_work$is_dmso])
  df_eta2_dmso <- map_dfr(dmso_plates_vec, function(p) {
    mat_p <- df_work |>
      filter(is_dmso, plate == p) |>
      select(all_of(feature_cols)) |>
      as.matrix()
    mat_r <- df_work |>
      filter(is_dmso, plate != p) |>
      select(all_of(feature_cols)) |>
      as.matrix()
    if (nrow(mat_p) == 0 || nrow(mat_r) == 0) {
      return(tibble())
    }
    tibble(
      compound = paste0("DMSO_", p),
      feature = feature_cols,
      eta2 = compute_eta2_two_groups(mat_p, mat_r),
      group = "DMSO"
    )
  })

  df_eff_dist <- bind_rows(df_eta2_cmpd, df_eta2_dmso) |>
    filter(is.finite(eta2), eta2 >= 0) |>
    mutate(group = factor(group, levels = c("DMSO", "compound")))

  p_eff_dist <- ggplot(df_eff_dist, aes(x = group, y = eta2, fill = group)) +
    geom_violin(trim = TRUE, alpha = 0.7, linewidth = 0.3) +
    geom_boxplot(
      width = 0.08,
      outlier.size = 0.3,
      fill = "white",
      linewidth = 0.3,
      alpha = 0.9
    ) +
    scale_fill_manual(
      values = c("DMSO" = "#d62728", "compound" = "#aec7e8"),
      guide = "none"
    ) +
    scale_y_log10() +
    labs(
      x = NULL,
      y = expression(eta^2 * " per feature (log scale)"),
      title = paste(
        "Effect size distribution: compounds vs DMSO null —",
        cycle_label
      ),
      subtitle = expression("DMSO null = per-plate leave-one-out " * eta^2)
    ) +
    theme_bw(base_size = 9) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )

  ggsave(
    paste0("supp_qc_effect_size_distribution", suffix_str, ".pdf"),
    plot = p_eff_dist,
    width = 100,
    height = 130,
    units = "mm"
  )

  message("  Top 10 features by signal-to-noise (eta2_compound / eta2_batch):")
  df_cv_feat_es |>
    slice_max(signal_to_noise, n = 10) |>
    select(
      feature_short,
      feature_type,
      eta2_batch,
      eta2_compound,
      signal_to_noise
    ) |>
    print()

  # -------------------------------------------------------------------------
  # 8. SUMMARY TABLES
  # -------------------------------------------------------------------------

  message("  Saving summary tables...")

  if (length(cluster_cols) > 0) {
    df_cluster_summary <- df_work |>
      group_by(plate, is_dmso) |>
      summarise(across(all_of(cluster_cols), mean), .groups = "drop") |>
      rename_with(~ str_c("mean_prop_", .), all_of(cluster_cols))
    df_cv_out <- df_cv |>
      left_join(df_cluster_summary, by = c("plate", "is_dmso"))
  } else {
    df_cv_out <- df_cv
  }
  write_csv(df_cv_out, paste0("supp_table_qc_summary", suffix_str, ".csv"))

  write_csv(
    df_cv_feat_es |>
      select(
        feature,
        feature_short,
        feature_type,
        cv_within_dmso,
        cv_total_dmso,
        cv_within_compound,
        cv_total_compound,
        cv_within,
        cv_total,
        eta2_batch,
        eta2_compound,
        signal_to_noise
      ) |>
      arrange(desc(signal_to_noise)),
    paste0("supp_table_feature_signal_noise", suffix_str, ".csv")
  )

  message("  Completed QC for suffix: ", suffix, "\n")
  return(df_cv_feat_es |> mutate(cycle = cycle_label))
}

# =============================================================================
# 3. RUN FOR EACH CYCLE
# =============================================================================

message(
  "\n============================================================================="
)
message("RUNNING QC METRICS FOR TUMOROID SCREEN DATASET")
message(
  "=============================================================================\n"
)

df_cycle1 <- compute_qc_metrics(df, "target", "cycle 1")
df_cycle3 <- compute_qc_metrics(df, "source", "cycle 3")

df_cv_feat_es <- bind_rows(
  if (!is.null(df_cycle1)) df_cycle1 else tibble(),
  if (!is.null(df_cycle3)) df_cycle3 else tibble()
)

if (nrow(df_cv_feat_es) > 0) {
  p_dmso <- df_cv_feat_es |>
    filter(is.finite(cv_within_dmso), is.finite(cv_total_dmso)) |>
    arrange(eta2_batch) |>
    ggplot(aes(x = cv_within_dmso, y = cv_total_dmso)) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey50"
    ) +
    geom_point(aes(color = eta2_batch, shape = cycle), size = 1, alpha = 0.8) +
    scale_color_viridis_c(
      option = "magma",
      name = expression(eta^2 * " batch"),
      limits = c(0, NA),
      direction = -1
    ) +
    scale_x_log10() +
    scale_y_log10() +
    coord_fixed() +
    labs(
      x = "CV within plates — DMSO (batch noise)",
      y = "CV total — all DMSO controls",
      title = "DMSO controls: batch noise vs total variance"
    ) +
    theme_bw(base_size = 6) +
    theme(
      legend.position = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  p_compound <- df_cv_feat_es |>
    filter(is.finite(cv_within_compound), is.finite(cv_total_compound)) |>
    arrange(eta2_compound) |>
    ggplot(aes(x = cv_within_compound, y = cv_total_compound)) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "grey50"
    ) +
    geom_point(
      aes(color = eta2_compound, shape = cycle),
      size = 1,
      alpha = 0.8
    ) +
    scale_color_viridis_c(
      option = "viridis",
      name = expression(eta^2 * " compound"),
      limits = c(0, NA)
    ) +
    scale_x_log10() +
    scale_y_log10() +
    coord_fixed() +
    labs(
      x = "CV within compound (replicate noise)",
      y = "CV total — all compounds",
      title = "Compounds: replicate noise vs total variance"
    ) +
    theme_bw(base_size = 6) +
    theme(
      legend.position = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  p_distance_dist <- df_cv_feat_es |>
    mutate(eta2_distance = eta2_compound - eta2_batch) |>
    ggplot(aes(x = eta2_distance, fill = cycle)) +
    geom_histogram(
      bins = 60,
      alpha = 0.7,
      position = "identity",
      linewidth = 0.1,
      color = "white"
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      color = "grey40",
      linewidth = 0.4
    ) +
    scale_fill_brewer(palette = "Set2") +
    labs(
      x = expression(eta^2 * " compound" ~ "-" ~ eta^2 * " batch"),
      y = "feature count",
      title = expression(
        "Distance: " * eta^2 * " compound" ~ "-" ~ eta^2 * " batch"
      ),
      subtitle = "Positive = more drug signal than batch noise"
    ) +
    theme_bw(base_size = 6) +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )

  p_combined <- (p_dmso | p_compound) / p_distance_dist

  ggsave(
    "supp_qc_cv_feature_scatter_effect_size.pdf",
    plot = p_combined,
    width = 180,
    height = 200,
    units = "mm"
  )
  ggsave(
    "supp_qc_cv_feature_scatter_effect_size.png",
    plot = p_combined,
    width = 180,
    height = 200,
    units = "mm"
  )
}

message("\nAll QC metrics completed!")
