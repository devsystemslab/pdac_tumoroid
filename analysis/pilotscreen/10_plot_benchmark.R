library(tidyverse)
library(patchwork)

cycles <- c('cycle_01', 'cycle_03')
screen <- 'pilotscreen'
dir_data <- 'data/processed'

df <- map(set_names(cycles), function(cycle) {
  dir_benchmark <- str_c(dir_data, screen, 'anndata', 'benchmarking', cycle, sep = '/')
  files <- list.files(dir_benchmark, pattern = '.csv')
  df <- map_df(str_c(dir_benchmark, files, sep = '/'), read_csv)
}) %>% bind_rows(.id = 'cycle')


plot_benchmark <- function(df, scores = c('gcs', 'clisis', 'nasw','mlami'), type='boxplot') {
  df_plot_scores <- df %>%
    select(c('type', 'cycle', scores)) %>%
    pivot_longer(values_to = 'score', names_to = 'measure', cols = scores)
  df_plot_scores_sum <- df_plot_scores %>%
     group_by(type, cycle, measure) %>%
     summarise(across(score, list(mean = \(x) mean(x, na.rm = TRUE),
                             sd = \(x) sd(x, na.rm = TRUE),
                             var = \(x) var(x, na.rm = TRUE),
                             median = \(x) median(x, na.rm = TRUE),
                             mad = \(x) mad(x, na.rm = TRUE))))
  if (type=='boxplot') {
    p <- ggplot(df_plot_scores, aes(x = type, y = score)) +
      geom_boxplot(outlier.alpha = 0) +
      facet_grid(measure ~ cycle) +
      theme_bw(base_size = 6) +
      ylim(c(0,1)) +
      xlab(NULL) +
      ylab(NULL) +
      theme(panel.grid = element_blank(),
            axis.text.x = element_text(angle = 45, hjust = 1))
  }
  if (type=='barplot'){
    p <- ggplot(df_plot_scores_sum,aes(x = type, y = score_mean, fill=type)) +
      geom_bar(stat='identity',col='black') +
      geom_errorbar(aes(ymin=score_mean-score_sd, ymax=score_mean+score_sd), width=.2,
                    position=position_dodge(.9)) +
      facet_grid(measure ~ cycle) +
      theme_bw(base_size = 6) +
      ylim(c(0, 1)) +
      scale_fill_brewer(palette = 'Set3') +
      xlab(NULL) +
      ylab(NULL) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }
  if (type=='tiles'){
    df_plot_scores_sum <- df_plot_scores_sum %>%
      mutate(label_mean = sprintf("%.2f", score_mean), measure_label = ifelse(str_detect(measure,'ari'),'ari','cnmi'))
    p <- ggplot(df_plot_scores_sum, aes(fill=score_mean, x=type, y=measure)) +
      geom_tile() +
      geom_text(aes(label=label_mean)) +
      coord_equal() +
      scale_x_discrete(expand = c(0,0)) +
      scale_y_discrete(expand = c(0,0)) +
      theme_bw(base_size = 6) +
      scale_fill_gradient2(midpoint = 0.5) +
      xlab(NULL) +
      ylab(NULL) +
      facet_grid(measure_label~ cycle)  +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))

  }

  return(p)
}
p1 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'cnmi')])
p2 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'ari')])
p3 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'gcs|clisis')])
p4 <- plot_benchmark(df, scores=colnames(df)[str_detect(colnames(df),'nasw|mlami')])

p1 + p2 + p3 + p4

p1 <- plot_benchmark(df, type='barplot')
p2 <- plot_benchmark(df, type='tiles', scores=colnames(df)[str_detect(colnames(df),'ari')])

p <- p1 / p2
dir_plots <- 'data/pilotscreen/plots'
ggsave(plot = p,
       filename = str_c(dir_plots, 'benchmarks.pdf', sep = '/'),
       dpi = 72, units = 'mm', width = 85*2, height = 125*2)
