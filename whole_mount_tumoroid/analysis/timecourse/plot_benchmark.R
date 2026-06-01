library(tidyverse)
library(patchwork)

screen <- "timecourse"
dir_data <- "/pstore/data/ihb-tumoroid/data/processed"
dir_benchmark <- str_c(dir_data, screen, "anndata", "benchmarking", sep = "/")
files <- list.files(dir_benchmark, pattern = ".csv")

df <- map_df(str_c(dir_benchmark, files, sep = "/"), read_csv) |> bind_rows(.id = "batch")
type_levels <- c("nuclei", "nuclei_msg", "imputed_nuclei", "imputed_neighbors", "phenocoder", "phenocoder_msg")
df$type <- factor(df$type, levels = type_levels)

plot_benchmark <- function(df, scores = c("gcs", "clisis", "nasw")) {
  df_plot_scores <- df |>
    select(c("type", scores)) |>
    pivot_longer(values_to = "score", names_to = "measure", cols = scores)
  p <- ggplot(df_plot_scores, aes(x = type, y = score)) +
    geom_boxplot(outlier.alpha = 0) +
    geom_jitter(width = 0.2, size = 0.1, color = "grey70", alpha = 0.25) +
    facet_wrap(~measure, nrow = 1) +
    theme_bw(base_size = 6) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  return(p)
}

plot_ari_conf_matrix <- function(df) {
  ari_cols <- colnames(df)[str_detect(colnames(df), "^ari_")]
  ari_levels <- str_c("ari", type_levels, sep = "_")
  df_plot_conf <- df |>
    select(c("type", all_of(ari_cols))) |>
    pivot_longer(values_to = "score", names_to = "ari_metric", cols = all_of(ari_cols)) |>
    group_by(type, ari_metric) |>
    summarise(mean_score = mean(score, na.rm = TRUE), .groups = "drop")
  df_plot_conf$ari_metric <- factor(df_plot_conf$ari_metric, levels = ari_levels)
  df_plot_conf$type <- factor(df_plot_conf$type, levels = type_levels)
  p <- ggplot(df_plot_conf, aes(x = type, y = ari_metric, fill = mean_score)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = round(mean_score, 2)), color = "white", size = 3) +
    scale_fill_distiller(palette = "RdBu") +
    theme_bw() +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      aspect.ratio = 1
    )
  return(p)
}


p1 <- plot_benchmark(df, scores = colnames(df)[str_detect(colnames(df), "cnmi")])
p2 <- plot_benchmark(df, scores = colnames(df)[str_detect(colnames(df), "ari")])
p3 <- plot_benchmark(df, scores = colnames(df)[str_detect(colnames(df), "gcs|clisis|nasw|mlami")])
p6 <- plot_ari_conf_matrix(df)
ggsave(plot = p3, filename = "benchmark_scores.pdf", width = 10, height = 5, dpi = 72, unit = "mm")
ggsave(plot = p6, filename = "ari_conf_matrix.pdf", width = 5, height = 5, dpi = 72, unit = "mm")
