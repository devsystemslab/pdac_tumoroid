library(tidyverse)
library(patchwork)

cycles <- c('cycle_01', 'cycle_03')
screen <- 'tumoroidscreen'
dir_data <- 'data/processed'

df <- map(set_names(cycles), function(cycle) {
  dir_benchmark <- str_c(dir_data, screen, 'anndata', 'benchmarking', cycle, sep = '/')
  files <- list.files(dir_benchmark, pattern = '.csv')
  df <- map_df(str_c(dir_benchmark, files, sep = '/'), read_csv)
}) %>% bind_rows(.id = 'cycle')


plot_benchmark <- function(df, scores = c('gcs', 'clisis', 'nasw')) {
  df_plot_scores <- df %>%
    select(c('type', 'cycle', scores)) %>%
    pivot_longer(values_to = 'score', names_to = 'measure', cols = scores)
  p <- ggplot(df_plot_scores, aes(x = type, y = score)) +
    geom_boxplot(outlier.alpha = 0) +
    facet_grid(measure ~ cycle, scales='free_y') +
    theme_bw() +
    theme(panel.grid = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1))
  return(p)
}
p1 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'cnmi')])
p2 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'ari')])
p3 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'gcs|clisis')])
p4 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'nasw')])

p1 + p2 + p3 + p4
