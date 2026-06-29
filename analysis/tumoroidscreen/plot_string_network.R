library(tidyverse)
library(tidygraph)
library(ggraph)
library(scales)
library(patchwork)


dir_analysis <- 'whole_mount_tumoroid/analysis/tumoroidscreen'
dir_plots <- 'data/processed/tumoroidscreen/plots'
df_genes_perturb <- readr::read_csv(str_c(dir_analysis,'tables','top100_genes_sorted.csv',sep='/'))
df_mc_cluster <- readr::read_tsv(str_c(dir_analysis,'string','string_MCL_clusters.tsv',sep='/'))
colnames(df_mc_cluster) <- str_replace(colnames(df_mc_cluster),' ','_')
df_nodes <- readr::read_tsv(str_c(dir_analysis,'string','string_network_coordinates_high.tsv',sep='/'))
colnames(df_nodes)[1] <- 'node'
df_nodes <- df_nodes %>%
  left_join(df_genes_perturb %>% select(gene:top_100) %>%
              mutate(gene=ifelse(gene=='COX1','MT-CO1', gene)) %>%
              rename(node=gene)) %>%
  left_join(df_mc_cluster %>% rename(node=protein_name)) %>%
  mutate(hex_color = ifelse(is.na(hex_color),'#b9b9b9',hex_color),
         cluster_number=ifelse(is.na(cluster_number),'isolated',as.character(mcl_cluster)),
         distance_sum_norm = ifelse(is.na(distance_sum_norm),mean(distance_sum_norm, na.rm = TRUE),distance_sum_norm)) %>%
  arrange(distance_sum_norm)
df_edges <- readr::read_tsv(str_c(dir_analysis,'string','string_interactions_short_high.tsv',sep='/'))
colnames(df_edges)[1:2] <- c('from','to')

g <- tbl_graph(df_nodes,df_edges, directed = TRUE)

mcl_cluster <- df_nodes %>% group_by(hex_color) %>%
  summarise(n=n()) %>%
  filter(n>2) %>%
  distinct(hex_color) %>% pull() %>% set_names()

list_subnetworks <- map(mcl_cluster, function(x){
  to_subgraph(g, hex_color == x, subset_by = "nodes")$subgraph
})

plot_network <- function(g, layout='string', label='all'){
  if (layout == 'string') {
    p <- ggraph(g, x=x_position, y=y_position)
  } else {
    p <- ggraph(g, layout=layout)
  }
  p <- p + geom_edge_fan(aes(alpha=rescale(combined_score)), show.legend = FALSE, edge_color='gray',strength = 9) +
  geom_node_point(aes(size=distance_sum_norm, fill=hex_color),shape = 21, color = 'black') +
  scale_size_continuous(range = c(5, 25)) +
  scale_alpha_continuous(range=c(0,0.05)) +
  scale_fill_identity() +
  theme_void() +
  coord_equal() +
  theme(legend.position = "right")
  if (label == 'all'){
    p <- p + geom_node_label(aes(label=node), repel = TRUE, size = 4)
  }
  if (label == 'filter'){
    p <- p + geom_node_label(aes(label=ifelse(distance_sum_norm > 2500 | node %in% c('TP53','BRAF',''), node,NA)), repel = TRUE, size = 4)
  }
  return(p)
}

plot_subnetwork <- function(g, layout='string'){
  if (layout == 'string') {
    p <- ggraph(g, x=x_position, y=y_position)
  } else {
    p <- ggraph(g, layout=layout)
  }
  p <- p + geom_edge_fan(aes(alpha=rescale(combined_score)), show.legend = FALSE, edge_color='gray',strength = 9) +
  #geom_node_point(aes(fill=hex_color), shape = 21, color = 'black', size=4) +
  geom_node_label(aes(label=node, fill=hex_color), repel = FALSE, size = 5) +
  scale_alpha_continuous(range=c(0,0.5)) +
  scale_fill_identity() +
  theme_void() +
  theme(legend.position = 'none')
  return(p)
}



p <- plot_network(g, layout = 'string', label='all')
ggsave(str_c(dir_plots,'string_network.pdf',sep='/'), p, width = 20, height = 10, dpi = 300, bg='white')
p_filter <- plot_network(g, layout = 'string', label='filter')
ggsave(str_c(dir_plots,'string_network_filter.pdf',sep='/'), p_filter, width = 18, height = 10, dpi = 72, bg='white')

p_sub <- map(list_subnetworks, function(x){
  p <- plot_subnetwork(x, layout = 'fr')
})
wrap_plots(p_sub)

ggsave(str_c(dir_plots,'string_network_sub.pdf',sep='/'), wrap_plots(p_sub), width = 30, height = 30, dpi = 72, bg='white')

p_sub
