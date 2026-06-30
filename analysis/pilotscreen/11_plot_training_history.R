library(tidyverse)
library(patchwork)
library(RColorBrewer)

dir_screen <- 'data/processed/pilotscreen'
dir_analysis <- 'analysis/pilotscreen'
dir_tensorboard <- str_c(dir_analysis,'plots','tensorboard', sep = '/')
datasets <- list.files(dir_tensorboard) %>% set_names()
# history plots - KL, reconstruction loss, epoch loss
read_history_files <- function(dataset, dir_data){
  files <- list.files(str_c(dir_data,dataset,sep='/'),pattern = '.csv', full.names = TRUE) %>%
    set_names(list.files(str_c(dir_data,dataset,sep='/'),pattern = '.csv'))
  names(files) <- str_split(names(files),'_') %>% map_chr(function(x){tail(x,3) %>% str_c(sep='_',collapse='_')}) %>% str_remove_all("^[0-9-]+|[0-9]")
  df <- map(files,readr::read_csv) %>% bind_rows(.id='dataset') %>%
    separate(col = dataset,into = c('split','tag','type') ,sep='-') %>%
    mutate(type=str_replace(type,'epoch_','') %>% str_remove('.csv'),
           split=str_remove_all(split,'_')) %>% select(-tag)
  colnames(df) <- colnames(df) %>% str_replace(' ','_') %>% str_to_lower()
  return(df)
}
df <- map(datasets,read_history_files,dir_data=dir_tensorboard) %>% bind_rows(.id='cycle') %>%
  mutate(cycle=str_replace(cycle,'pilotscreen_cycle','cycle ') %>% str_to_sentence(),
         type=str_replace(type,'_', ' ') %>% str_to_sentence())
df$type <- factor(df$type, levels = c('Loss','Reconstruction loss','Kl loss','Learning rate'))

p <- ggplot(df %>% filter(type!='Learning rate'), aes(x=step,y=value, col=split)) +
  geom_line(linewidth=0.5) +
  facet_wrap(cycle~type, scales = 'free') +
  scale_color_brewer(type='qual') +
  theme_bw(base_size = 6) +
  theme(panel.grid.minor = element_blank(),
        legend.position = 'bottom') +
  ylab(NULL) +
  xlab('Epoch')
ggsave(str_c(dir_screen,'plots','train_history.pdf', sep='/'), p, dpi=72, width=90, height=90, units='mm')
