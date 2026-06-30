library(anndata)
library(tidyverse)
library(patchwork)
library(ggrepel)

source(
  "analysis/pilotscreen/utils.R"
)
setwd(
  "analysis/pilotscreen/plots"
)

# =============================================================================
# 1. DATA LOADING
# =============================================================================

message("Loading data...")

file <- "data/pilotscreen/anndata/mdata_org_combined.h5mu"
mdata <- read_mdata(file)

df <- mdata["phenocoder_combined"]$obs |>
  as_tibble() |>
  mutate(
    compound = as.character(compound) |> str_replace("_", " "),
    conc = as.character(conc) |> str_replace("_", " "),
    timepoint = as.character(timepoint) |>
      str_replace("_", " ") |>
      str_to_sentence()
  ) |>
  mutate(
    compound = ifelse(conc == "0 µM", "DMSO", compound)
  ) |>
  mutate(
    timepoint = factor(timepoint, levels = c("Day 4", "Day 7", "Day 11")),
    conc = factor(conc, levels = c("0 µM", "1 µM", "5 µM", "10 µM") |> rev())
  ) |>
  mutate(
    condition = str_c(compound, conc, timepoint, sep = "_")
  ) |>
  bind_cols(mdata["phenocoder_combined"]$layers["raw"] |> as_tibble())

# color palette
colors_compound <- c(
  `Ac-Gly-BoroPro` = "#1f77b4",
  Bortezomib = "#ff7f0e",
  `BTT-3033` = "#279e68",
  DMSO = "#d62728",
  Erlotinib = "#aa40fc",
  Gemcitabine = "#8c564b",
  Ilomastat = "#e377c2",
  `LGK-974` = "#b5bd61",
  Linsitinib = "#17becf",
  Paclitaxel = "#aec7e8",
  `PF-562271` = "#ffbb78",
  SN38 = "#98df8a",
  T0070907 = "#ff9896",
  Trametinib = "#c5b0d5",
  VER155008 = "#c49c94"
)

# =============================================================================
# 2. QUALITY CONTROL FUNCTION
# =============================================================================

compute_qc_metrics <- function(df, suffix, colors_compound) {
  message(paste("\n========== Processing", suffix, "==========\n"))

  # Identify columns with the given suffix
  all_cols <- colnames(df)
  cell_count_col <- paste0(
    "cell_count",
    if (suffix != "") paste0("_", suffix) else ""
  )
  volume_col <- paste0(
    "phenocoder_stat_volume_chull",
    if (suffix != "") paste0("_", suffix) else "$"
  )
  cluster_cols <- all_cols[str_detect(
    all_cols,
    paste0(
      "^phenocoder_\\d+",
      if (suffix != "") paste0("_", suffix) else "$",
      "$"
    )
  )]
  cluster_cols_msg <- all_cols[str_detect(
    all_cols,
    paste0(
      "^phenocoder_msg_\\d+",
      if (suffix != "") paste0("_", suffix) else "$",
      "$"
    )
  )]

  message(
    "Found cluster proportion columns: ",
    paste(cluster_cols, collapse = ", ")
  )
  message(
    "Found msg cluster columns: ",
    paste(cluster_cols_msg, collapse = ", ")
  )

  # Check if required columns exist
  if (!cell_count_col %in% colnames(df)) {
    message(
      "Column ",
      cell_count_col,
      " not found. Skipping QC for suffix: ",
      suffix
    )
    return(NULL)
  }

  # Create working dataframe with renamed columns for consistency
  df_work <- df |>
    rename(
      cell_count = all_of(cell_count_col),
      volume = all_of(volume_col)
    ) |>
    mutate(density = cell_count / volume)

  suffix_str <- if (suffix != "") paste0("_", suffix) else ""

  # =============================================================================
  # 3. SAMPLE SIZES
  # =============================================================================

  message("Computing sample sizes...")

  df_n <- df_work |>
    count(compound, conc, timepoint, condition) |>
    arrange(compound, conc, timepoint)

  p_n <- ggplot(
    df_n,
    aes(x = interaction(conc, timepoint, sep = "\n"), y = n, fill = compound)
  ) +
    geom_col(color = "black", linewidth = 0.2) +
    geom_text(aes(label = n), vjust = -0.3, size = 2.5) +
    facet_wrap(~compound, scales = "free_x", nrow = 3) +
    scale_fill_manual(values = colors_compound, name = "Compound") +
    labs(x = NULL, y = "n (organoids)", title = "Sample size per condition") +
    theme_bw(base_size = 9) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )

  ggsave(
    paste0("supp_qc_sample_sizes", suffix_str, ".pdf"),
    plot = p_n,
    width = 280,
    height = 250,
    units = "mm"
  )

  # =============================================================================
  # 4. BASIC METRICS BOXPLOTS
  # =============================================================================

  message("Plotting basic metrics...")

  df_metrics <- df_work |>
    select(plate_id, compound, conc, timepoint, condition, cell_count, volume, density) |>
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

  # Remove outliers from density using IQR method
  df_metrics <- df_metrics |>
    group_by(metric) |>
    mutate(
      Q1 = quantile(value, 0.25, na.rm = TRUE),
      Q3 = quantile(value, 0.75, na.rm = TRUE),
      IQR = Q3 - Q1,
      lower_bound = Q1 - 1.5 * IQR,
      upper_bound = Q3 + 0.75 * IQR,
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
    aes(
      x = interaction(conc, timepoint, sep = " | "),
      y = value,
      fill = compound
    )
  ) +
    geom_boxplot(outlier.size = 0.5, linewidth = 0.3) +
    facet_grid(metric ~ compound, scales = "free") +
    scale_fill_manual(values = colors_compound, name = "Compound") +
    labs(
      x = NULL,
      y = NULL,
      title = "Basic organoid metrics across conditions"
    ) +
    theme_bw(base_size = 9) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, size = 5),
      strip.text = element_text(face = "bold", size = 8),
      legend.position = "bottom"
    )

  p_metrics_dmso <- ggplot(
    df_metrics %>% filter(compound == "DMSO", metric != "Density (cells/volume)"),
    aes(
      x = plate_id,
      group = plate_id,
      y = value
    )
  ) +
    facet_wrap(~metric, scales = "free_y") +
    geom_boxplot(outlier.alpha=0) +
    geom_jitter() +
    labs(
      x = NULL,
      y = NULL
    ) +
    ylim(c(0,NA)) +
    theme_bw(base_size = 6) +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, size = 5),
      strip.text = element_text(face = "bold", size = 8),
      legend.position = "bottom",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    )

  # T-tests: plate 004 vs 005 for each metric
  df_dmso <- df_metrics %>% filter(compound == "DMSO")

  ttest_results <- list()
  metrics_to_test <- c(
    "Cell count",
    "Volume (convex hull)",
    "Density (cells / volume)"
  )

  p_values <- c()

  message("\nT-test results: Plate 004 vs 005 (DMSO controls)")
  message("=", paste(rep("-", 60), collapse = ""))

  for (met in metrics_to_test) {
    data_004 <- df_dmso %>%
      filter(metric == met, plate_id == "004") %>%
      pull(value) %>%
      na.omit()

    data_005 <- df_dmso %>%
      filter(metric == met, plate_id == "005") %>%
      pull(value) %>%
      na.omit()

    # Filter outliers: keep values between 1st and 99th percentiles
    p01_004 <- quantile(data_004, 0.01, na.rm = TRUE)
    p99_004 <- quantile(data_004, 0.99, na.rm = TRUE)
    data_004 <- data_004[data_004 >= p01_004 & data_004 <= p99_004]

    p01_005 <- quantile(data_005, 0.01, na.rm = TRUE)
    p99_005 <- quantile(data_005, 0.99, na.rm = TRUE)
    data_005 <- data_005[data_005 >= p01_005 & data_005 <= p99_005]

    if (length(data_004) > 0 && length(data_005) > 0) {
      t_test <- t.test(data_004, data_005)
      mw_test <- wilcox.test(data_004, data_005)
      ttest_results[[met]] <- list(t_test = t_test, mw_test = mw_test)
      p_values <- c(p_values, t_test$p.value)

      message(sprintf(
        "\n%s:",
        met
      ))
      message(sprintf(
        "  Plate 004: n=%d, mean=%.3f, sd=%.3f",
        length(data_004), mean(data_004), sd(data_004)
      ))
      message(sprintf(
        "  Plate 005: n=%d, mean=%.3f, sd=%.3f",
        length(data_005), mean(data_005), sd(data_005)
      ))
      message(sprintf(
        "  t-test: t=%.3f, p=%.4f",
        t_test$statistic, t_test$p.value
      ))
      message(sprintf(
        "  Mann-Whitney U: U=%.3f, p=%.4f",
        mw_test$statistic, mw_test$p.value
      ))
    }
  }

  # Apply Bonferroni correction for multiple testing
  p_values_corrected <- p.adjust(p_values, method = "bonferroni")

  # Collect Mann-Whitney p-values
  mw_p_values <- c()
  for (met in metrics_to_test) {
    if (met %in% names(ttest_results) && !is.null(ttest_results[[met]]$mw_test)) {
      mw_p_values <- c(mw_p_values, ttest_results[[met]]$mw_test$p.value)
    }
  }
  mw_p_values_corrected <- p.adjust(mw_p_values, method = "bonferroni")

  message("\nBonferroni-corrected p-values (t-test):")
  message("=", paste(rep("-", 60), collapse = ""))
  for (i in seq_along(metrics_to_test)) {
    sig_marker <- ifelse(p_values_corrected[i] < 0.05, "*", "")
    message(sprintf(
      "%s: adjusted p = %.4f %s",
      metrics_to_test[i],
      p_values_corrected[i],
      sig_marker
    ))
  }
  message("\nBonferroni-corrected p-values (Mann-Whitney U):")
  message("=", paste(rep("-", 60), collapse = ""))
  for (i in seq_along(metrics_to_test)) {
    sig_marker <- ifelse(mw_p_values_corrected[i] < 0.05, "*", "")
    message(sprintf(
      "%s: adjusted p = %.4f %s",
      metrics_to_test[i],
      mw_p_values_corrected[i],
      sig_marker
    ))
  }
  message("=", paste(rep("-", 60), collapse = ""))


  ggsave(
    paste0("supp_qc_basic_metrics", suffix_str, ".pdf"),
    plot = p_metrics,
    width = 380,
    height = 200,
    units = "mm"
  )
  ggsave(
    paste0("supp_qc_basic_metrics_dmso", suffix_str, ".pdf"),
    plot = p_metrics_dmso,
    width = 9,
    height = 6,
    dpi=72
  )

  # =============================================================================
  # 5. CLUSTER PROPORTIONS
  # =============================================================================

  message("Plotting cluster proportions...")

  if (length(cluster_cols) > 0) {
    df_clusters <- df_work |>
      select(
        compound,
        conc,
        timepoint,
        condition,
        cell_count,
        all_of(cluster_cols)
      ) |>
      group_by(compound, conc, timepoint, condition) |>
      mutate(across(all_of(cluster_cols), ~ . / cell_count)) |>
      summarise(across(all_of(cluster_cols), mean), .groups = "drop") |>
      pivot_longer(
        cols = all_of(cluster_cols),
        names_to = "cluster",
        values_to = "proportion"
      ) |>
      mutate(
        cluster = str_replace(cluster, "phenocoder_", "cluster "),
        cluster = str_replace(cluster, paste0("_", suffix), "")
      )

    p_clusters <- ggplot(
      df_clusters,
      aes(
        x = interaction(conc, timepoint, sep = "\n"),
        y = proportion,
        fill = cluster
      )
    ) +
      geom_col(position = "stack", color = "black", linewidth = 0.1) +
      facet_wrap(~compound, scales = "free_x", nrow = 3) +
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
      paste0("supp_qc_cluster_proportions", suffix_str, ".pdf"),
      plot = p_clusters,
      width = 280,
      height = 250,
      units = "mm"
    )
  } else {
    message("  no cluster proportion columns found for suffix: ", suffix)
  }

  # =============================================================================
  # 6. MESSAGE-PASSED CLUSTER PROPORTIONS
  # =============================================================================

  message("Plotting message-passed cluster proportions...")

  if (length(cluster_cols_msg) > 0) {
    df_clusters <- df_work |>
      select(
        compound,
        conc,
        timepoint,
        condition,
        cell_count,
        all_of(cluster_cols_msg)
      ) |>
      group_by(compound, conc, timepoint, condition) |>
      mutate(across(all_of(cluster_cols_msg), ~ . / cell_count)) |>
      summarise(across(all_of(cluster_cols_msg), mean), .groups = "drop") |>
      pivot_longer(
        cols = all_of(cluster_cols_msg),
        names_to = "cluster",
        values_to = "proportion"
      ) |>
      mutate(
        cluster = str_replace(cluster, "phenocoder_msg_", "cluster "),
        cluster = str_replace(cluster, paste0("_", suffix), "")
      )

    p_clusters <- ggplot(
      df_clusters,
      aes(
        x = interaction(conc, timepoint, sep = "\n"),
        y = proportion,
        fill = cluster
      )
    ) +
      geom_col(position = "stack", color = "black", linewidth = 0.1) +
      facet_wrap(~compound, scales = "free_x", nrow = 3) +
      scale_fill_brewer(palette = "Set2", name = "phenocoder\ncluster") +
      scale_x_discrete(expand = c(0, 0)) +
      scale_y_continuous(expand = c(0, 0)) +
      labs(
        x = NULL,
        y = "mean proportion",
        title = "cvae message-passed cluster composition across conditions"
      ) +
      theme_bw(base_size = 9) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        strip.text = element_text(face = "bold"),
        legend.position = "right"
      )

    ggsave(
      paste0("supp_qc_msg_cluster_proportions", suffix_str, ".pdf"),
      plot = p_clusters,
      width = 280,
      height = 250,
      units = "mm"
    )
  } else {
    message(
      "  no message-passed cluster proportion columns found for suffix: ",
      suffix
    )
  }

  # =============================================================================
  # 7. COEFFICIENT OF VARIATION
  # =============================================================================

  message("Computing coefficient of variation...")

  df_cv <- df_work |>
    group_by(compound, conc, timepoint, condition) |>
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
      mean_density = mean(density, na.rm = TRUE),
      sd_density = sd(density, na.rm = TRUE),
      .groups = "drop"
    )

  df_cv_long <- df_cv |>
    select(compound, conc, timepoint, condition, starts_with("cv_")) |>
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
    aes(x = interaction(conc, timepoint, sep = " | "), y = cv, fill = compound)
  ) +
    geom_col(color = "black", linewidth = 0.2) +
    facet_grid(metric ~ compound, scales = "free_x") +
    scale_fill_manual(values = colors_compound, name = "compound") +
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
      axis.text.x = element_text(angle = 90, hjust = 1, size = 5),
      strip.text = element_text(face = "bold", size = 8),
      legend.position = "bottom"
    )

  ggsave(
    paste0("supp_qc_cv_summary", suffix_str, ".pdf"),
    plot = p_cv,
    width = 380,
    height = 180,
    units = "mm"
  )

  # =============================================================================
  # 8. FEATURE CV SCATTER: CV_within vs CV_total + ANOVA
  # =============================================================================
  # CV_total  = SD / mean across ALL organoids
  # CV_within = mean of per-condition CVs
  #
  # Features far above the diagonal have variation primarily explained by
  # experimental conditions (condition-informative).
  #
  # ANOVA: feature ~ compound + conc + timepoint
  # Extract compound F-statistic to isolate drug identity effect from
  # concentration and timepoint.
  # =============================================================================

  message("Computing per-feature CV scatter + ANOVA...")

  # get all raw feature columns that match the suffix
  all_raw_cols <- colnames(
    mdata["phenocoder_combined"]$layers["raw"] |> as_tibble()
  )
  if (suffix != "") {
    feature_cols <- all_raw_cols[str_detect(
      all_raw_cols,
      paste0("_", suffix, "$")
    )]
  } else {
    feature_cols <- all_raw_cols
  }

  # exclude columns that were renamed in df_work (cell_count_X -> cell_count, volume_col -> volume)
  renamed_originals <- c(cell_count_col, volume_col)
  feature_cols <- setdiff(feature_cols, renamed_originals)

  message("  Found ", length(feature_cols), " features for suffix: ", suffix)

  if (length(feature_cols) > 0) {
    # CV_total: across all organoids
    cv_total <- df_work |>
      summarise(across(
        all_of(feature_cols),
        ~ sd(., na.rm = TRUE) / abs(mean(., na.rm = TRUE))
      )) |>
      pivot_longer(everything(), names_to = "feature", values_to = "cv_total")

    # CV_within: mean of per-condition CVs
    cv_within <- df_work |>
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

    # --- ANOVA per feature: feature ~ compound + conc + timepoint ---
    message("  Running ANOVA (feature ~ compound + conc + timepoint)...")

    anova_results <- tibble(
      feature = feature_cols,
      f_stat_compound = NA_real_,
      p_value_compound = NA_real_
    )

    for (i in seq_along(feature_cols)) {
      feat <- feature_cols[i]
      vals <- df_work[[feat]]

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
          fit <- aov(vals ~ df_work$compound + df_work$conc + df_work$timepoint)
          s <- summary(fit)[[1]]
          # compound is the 1st term
          list(f = s[["F value"]][1], p = s[["Pr(>F)"]][1])
        },
        error = function(e) list(f = NA_real_, p = NA_real_)
      )

      anova_results$f_stat_compound[i] <- res$f
      anova_results$p_value_compound[i] <- res$p
    }

    anova_results <- anova_results |>
      mutate(p_adj_compound = p.adjust(p_value_compound, method = "bonferroni"))

    # --- Build combined dataframe ---
    # strip suffix for display
    feat_suffix_pattern <- if (suffix != "") paste0("_", suffix, "$") else ""

    df_cv_feat <- cv_total |>
      left_join(cv_within, by = "feature") |>
      left_join(anova_results, by = "feature") |>
      filter(is.finite(cv_total) & is.finite(cv_within)) |>
      mutate(
        feature_short = if (suffix != "") {
          str_replace(feature, paste0("_", suffix, "$"), "")
        } else {
          feature
        },
        feature_type = case_when(
          str_detect(feature, "interaction") ~ "interaction",
          str_detect(feature, "centrality") ~ "centrality",
          str_detect(feature, "nhood_z") ~ "neighborhood enrichment",
          str_detect(feature, "degree") ~ "degree",
          str_detect(feature, "stat_z") ~ "spatial autocorrelation",
          str_detect(feature, "^phenocoder_\\d+") ~ "cluster proportion",
          str_detect(
            feature,
            "^phenocoder_msg_\\d+"
          ) ~ "cluster proportion (msg)",
          str_detect(feature, "volume|cell_count|density") ~ "morphological",
          TRUE ~ "other"
        ),
        ratio = cv_total / cv_within,
        neg_log10_p = -log10(pmax(p_adj_compound, 1e-300)),
        anova_sig = case_when(
          is.na(p_adj_compound) ~ "NA",
          p_adj_compound < 0.001 ~ "***",
          p_adj_compound < 0.01 ~ "**",
          p_adj_compound < 0.05 ~ "*",
          TRUE ~ "ns"
        )
      )

    # identify features to label
    top_informative <- df_cv_feat |>
      slice_max(ratio, n = 10) |>
      pull(feature)

    top_anova <- df_cv_feat |>
      filter(!is.na(f_stat_compound)) |>
      slice_max(f_stat_compound, n = 10) |>
      pull(feature)

    top_labels <- unique(c(top_informative, top_anova))

    # --- Plot A: colored by feature type ---
    p_cv_type <- ggplot(df_cv_feat, aes(x = cv_within, y = cv_total)) +
      geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed",
        color = "grey50"
      ) +
      geom_point(aes(color = feature_type), size = 1.5, alpha = 0.7) +
      scale_y_log10() +
      scale_x_log10() +
      geom_text_repel(
        data = df_cv_feat |> filter(feature %in% top_labels),
        aes(label = feature_short),
        size = 1.8,
        max.overlaps = 20,
        segment.size = 0.2,
        segment.color = "grey60",
        min.segment.length = 0
      ) +
      scale_color_brewer(palette = "Set1", name = "Feature type") +
      labs(
        x = "CV within conditions (mean per-condition CV)",
        y = "CV total (across all organoids)",
        title = "Feature informativeness by type"
      ) +
      theme_bw(base_size = 9) +
      theme(legend.position = "right")

    # --- Plot B: colored by ANOVA compound significance ---
    p_cv_anova <- ggplot(df_cv_feat, aes(x = cv_within, y = cv_total)) +
      geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed",
        color = "grey50"
      ) +
      geom_point(aes(color = neg_log10_p), size = 1.5, alpha = 0.7) +
      scale_y_log10() +
      scale_x_log10() +
      geom_text_repel(
        data = df_cv_feat |> filter(feature %in% top_labels),
        aes(label = feature_short),
        size = 1.8,
        max.overlaps = 20,
        segment.size = 0.2,
        segment.color = "grey60",
        min.segment.length = 0
      ) +
      scale_color_viridis_c(
        option = "inferno",
        name = expression(-log[10](p[adj])),
        direction = -1
      ) +
      labs(
        x = "CV within conditions (mean per-condition CV)",
        y = "CV total (across all organoids)",
        title = "Feature informativeness by compound ANOVA (feature ~ compound + conc + timepoint)"
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
      geom_hline(
        yintercept = -log10(0.05),
        linetype = "dashed",
        color = "grey50"
      ) +
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

    p_cv_combined <- (p_cv_type | p_cv_anova) /
      (p_annova_rank | p_box_plot_annova)

    ggsave(
      paste0("supp_qc_cv_feature_scatter", suffix_str, ".pdf"),
      plot = p_cv_combined,
      width = 220,
      height = 350,
      units = "mm"
    )

    # save ANOVA table
    write_csv(
      df_cv_feat |>
        select(
          feature,
          feature_short,
          feature_type,
          cv_within,
          cv_total,
          ratio,
          f_stat_compound,
          p_value_compound,
          p_adj_compound,
          anova_sig
        ) |>
        arrange(desc(f_stat_compound)),
      paste0("supp_table_feature_anova", suffix_str, ".csv")
    )

    message("  ANOVA results saved. Top 10 features by compound F-statistic:")
    df_cv_feat |>
      filter(!is.na(f_stat_compound)) |>
      slice_max(f_stat_compound, n = 10) |>
      select(
        feature_short,
        f_stat_compound,
        p_adj_compound,
        anova_sig,
        cv_total,
        cv_within,
        ratio
      ) |>
      print()

    # =============================================================================
    # 8B. EFFECT SIZES: eta^2, omega^2, Cohen's f
    # =============================================================================
    # With N in the thousands, p-values are dominated by sample size. Effect
    # sizes answer the question we actually care about: how much of each
    # feature's variance is explained by compound?
    #
    #   eta^2    = SSB / SST                    (proportion variance explained)
    #   omega^2  = (SSB - (k-1)*MSW) / (SST + MSW)
    #              bias-corrected; small/neg for null effects
    #   Cohen's f = sqrt(eta^2 / (1 - eta^2))    (conventional thresholds:
    #              0.10 small, 0.25 medium, 0.40 large)
    #
    # Note: for one-way ANOVA, eta^2 == partial eta^2 (only one factor).
    # =============================================================================

    message("  Computing effect sizes (eta^2, omega^2, Cohen's f)...")

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
      res <- effect_size_one(df_work[[feat]], df_work$compound)
      effect_results$eta2[i] <- res$eta2
      effect_results$omega2[i] <- res$omega2
      effect_results$cohen_f[i] <- res$cohen_f
    }

    df_cv_feat_es <- effect_results |>
      left_join(cv_total, by = "feature") |>
      left_join(cv_within, by = "feature") |>
      mutate(
        feature_short = if (suffix != "") {
          str_replace(feature, paste0("_", suffix, "$"), "")
        } else {
          feature
        },
        feature_type = case_when(
          str_detect(feature_short, "interaction") ~ "interaction",
          str_detect(
            feature,
            "degree|closeness|centrality|stat_mean|stat_std"
          ) ~ "node connectivity",
          str_detect(feature_short, "nhood_z") ~ "neighborhood enrichment",
          str_detect(feature_short, "stat_z") ~ "spatial autocorrelation",
          str_detect(
            feature,
            "chulls|chull|n_pts|distance_center"
          ) ~ "convex hull stats",
          str_detect(feature_short, "^phenocoder_\\d+") ~ "cluster proportion",
          str_detect(
            feature_short,
            "^phenocoder_msg_\\d+"
          ) ~ "cluster proportion",
          str_detect(
            feature_short,
            "^phenocoder_msg_\\d+"
          ) ~ "cluster proportion (msg)",
          str_detect(
            feature_short,
            "volume|cell_count|density"
          ) ~ "morphological",
          TRUE ~ "other"
        ),
        kernel_size = as.integer(
          str_extract(feature_short, "(?<=_)(25|50|100|150)$")
        ),
        kernel_size = ifelse(
          feature_type == "convex hull stats",
          100,
          kernel_size
        ),
        scope = if_else(is.na(kernel_size), "whole organoid", "local")
      ) |>
      mutate(
        kernel_size_f = factor(
          kernel_size,
          levels = c(25, 50, 100, 150),
          exclude = NULL
        ) |>
          fct_na_value_to_level("whole organoid")
      )
    return(df_cv_feat_es)
    # --- Plot A: CV scatter colored by omega^2 ---
    p_cv_omega <- ggplot(df_cv_feat_es, aes(x = cv_within, y = cv_total)) +
      geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed",
        color = "grey50"
      ) +
      geom_point(aes(color = omega2), size = 1.5, alpha = 0.8) +
      scale_color_viridis_c(
        option = "viridis",
        name = expression(omega^2),
        limits = c(0, NA)
      ) +
      labs(
        x = "CV within conditions (mean per-condition CV)",
        y = "CV total (across all organoids)"
      ) +
      scale_x_log10() +
      scale_y_log10() +
      theme_bw(base_size = 9) +
      theme(legend.position = "right")

    p_cv_type_es <- ggplot(df_cv_feat_es, aes(x = cv_within, y = cv_total)) +
      geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed",
        color = "grey50"
      ) +
      geom_point(aes(color = feature_type), size = 1.5, alpha = 0.8) +
      scale_color_brewer(palette = "Set1", name = "Feature type") +
      labs(
        x = "CV within conditions (mean per-condition CV)",
        y = "CV total (across all organoids)"
      ) +
      scale_x_log10() +
      scale_y_log10() +
      theme_bw(base_size = 9) +
      theme(legend.position = "right")

    p_omega_box <- ggplot(
      df_cv_feat_es,
      aes(x = kernel_size_f, y = omega2, fill = kernel_size_f)
    ) +
      geom_boxplot(
        position = position_dodge(preserve = "single"),
        outlier.size = 0.5
      ) +
      facet_wrap(~feature_type, nrow = 1, scales = "free_x") +
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

    p_cv_es_combined <- (p_cv_omega | p_cv_type_es) / p_omega_box

    ggsave(
      paste0("supp_qc_cv_feature_scatter_effect_size", suffix_str, ".pdf"),
      plot = p_cv_es_combined,
      width = 220,
      height = 220,
      units = "mm"
    )

    ggsave(
      paste0("supp_qc_cv_feature_scatter_effect_size", suffix_str, ".png"),
      plot = p_cv_es_combined,
      width = 220,
      height = 220,
      units = "mm"
    )

    message("  Effect size analysis completed.")
  } else {
    message(
      "  No features found for suffix: ",
      suffix,
      ", skipping CV scatter."
    )
  }

  # =============================================================================
  # 9. SUMMARY TABLE
  # =============================================================================

  message("Saving summary table...")

  if (length(cluster_cols) > 0) {
    df_cluster_summary <- df_work |>
      group_by(compound, conc, timepoint, condition) |>
      summarise(across(all_of(cluster_cols), mean), .groups = "drop") |>
      rename_with(~ str_c("mean_prop_", .), all_of(cluster_cols))

    df_cv_full <- df_cv |>
      left_join(
        df_cluster_summary,
        by = c("compound", "conc", "timepoint", "condition")
      )
  } else {
    df_cv_full <- df_cv
  }

  write_csv(df_cv_full, paste0("supp_table_qc_summary", suffix_str, ".csv"))

  message("Completed QC for suffix: ", suffix, "\n")
}

# =============================================================================
# 10. RUN QC FOR EACH CELL TYPE
# =============================================================================

message(
  "\n============================================================================="
)
message("RUNNING QC METRICS FOR pilotscreen DATASET")
message(
  "=============================================================================\n"
)

# Process target cells
df_target <- compute_qc_metrics(df, "target", colors_compound) %>% mutate(cycle = "target")

# Process source cells
df_source <- compute_qc_metrics(df, "source", colors_compound) %>% mutate(cycle = "source")
df_cv_feat_es <- bind_rows(df_target, df_source)

# --- Plot A: CV scatter colored by omega^2 ---
p_cv_omega <- ggplot(df_cv_feat_es %>% arrange(eta2), aes(x = cv_within, y = cv_total)) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "grey50"
  ) +
  geom_point(aes(color = eta2, shape = cycle), size = 1) +
  scale_color_viridis_c(
    option = "viridis",
    name = expression(eta^2),
    limits = c(0, NA)
  ) +
  labs(
    x = "CV within conditions (mean per-condition CV)",
    y = "CV total (across all organoids)"
  ) +
  scale_x_log10() +
  scale_y_log10() +
  coord_fixed() +
  theme_bw(base_size = 6) +
  theme(legend.position = "right", panel.grid.major = element_blank(), panel.grid.minor = element_blank())

p_cv_type_es <- ggplot(df_cv_feat_es %>% arrange(desc(feature_type)), aes(x = cv_within, y = cv_total)) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "grey50"
  ) +
  geom_point(aes(color = feature_type, shape = cycle), size = 1, alpha = 0.8) +
  scale_color_brewer(palette = "Set1", name = "Feature type") +
  labs(
    x = "CV within conditions (mean per-condition CV)",
    y = "CV total (across all organoids)"
  ) +
  scale_x_log10() +
  scale_y_log10() +
  coord_fixed() +
  theme_bw(base_size = 6) +
  theme(legend.position = "right", panel.grid.major = element_blank(), panel.grid.minor = element_blank())

p_omega_box <- ggplot(
  df_cv_feat_es,
  aes(x = kernel_size_f, y = eta2, fill = kernel_size_f)
) +
  geom_boxplot(
    position = position_dodge(preserve = "single"),
    outlier.size = 0.5
  ) +
  facet_wrap(~feature_type, nrow = 1, scales = "free_x") +
  scale_fill_brewer(palette = "Set2", name = "Kernel size") +
  labs(
    x = "Feature type",
    y = expression(eta^2),
    title = expression(eta^2 * " by feature type and kernel size")
  ) +
  theme_bw(base_size = 6) +
  theme(
    legend.position = "right",
    panel.grid.major = element_blank(), panel.grid.minor = element_blank()
  )

p_cv_es_combined <- (p_cv_omega | p_cv_type_es) / p_omega_box
p_cv_es_combined <- p_cv_es_combined + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
ggsave(
  paste0("supp_qc_cv_feature_scatter_effect_size.pdf"),
  plot = p_cv_es_combined,
  width = 90 * 2,
  height = 90 * 2,
  units = "mm"
)
ggsave(
  paste0("supp_qc_cv_feature_scatter_effect_size.png"),
  plot = p_cv_es_combined,
  width = 90 * 2,
  height = 90 * 2,
  units = "mm"
)
message("\nAll QC metrics completed!")
