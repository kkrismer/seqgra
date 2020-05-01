#!/usr/bin/env Rscript

library(methods)

# hack apparently necessary on Windows if Sys.getenv("HOME") returns
# C:/Users/[User] instead of C:/Users/[User]\Documents
user_site_path <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_site_path)) {
  warning(paste0("R_LIBS_USER environment variable set incorrectly"))
  user_site_path <- gsub("//", "/", gsub("\\", "/", user_site_path, fixed = TRUE), fixed = TRUE)
  home_path <- gsub("//", "/", gsub("\\", "/", Sys.getenv("HOME"), fixed = TRUE), fixed = TRUE)
  user_site_path <- gsub(home_path, "", user_site_path)
  user_site_path <- gsub("//", "/", paste0(home_path, "/Documents/", user_site_path), fixed = TRUE)
  .libPaths(c(.libPaths(), user_site_path))
}

library(ggplot2)
library(scales)

plot_agreement <- function(input_file_name, output_file_name, title = NULL) {
  # df with example, position, group, label, precision, recall, 
  # specificity, f1, n
  # where precision, recall, specificity, f1 are mean values per label,
  # and n is the number of examples per label
  # if df contains value column, switch to non-thresholded mode
  example <- position <- label <- group <- NULL
  
  df <- read.table(input_file_name, header = TRUE, sep = "\t",
                   stringsAsFactors = FALSE)
  
  thresholded_mode <- !("value" %in% colnames(df))
  
  df$label <- paste0("label: ", df$label,
                     " | precision = ", round(df$precision, digits = 3),
                     ", recall (sensitivity) = ",
                     round(df$recall, digits = 3),
                     ", specificity = ", round(df$specificity, digits = 3), 
                     ", F1 = ", round(df$f1, digits = 3),
                     ", n = ", df$n)
  df$example <- as.factor(df$example)
  
  if (thresholded_mode) {
    levels <- c("TP (grammar position, model position)",
                "FN (grammar position, no model position)",
                "FP (background position, model position)",
                "TN (background position, no model position)")
    df$group[toupper(df$group) == "TP"] <- levels[1]
    df$group[toupper(df$group) == "FN"] <- levels[2]
    df$group[toupper(df$group) == "FP"] <- levels[3]
    df$group[toupper(df$group) == "TN"] <- levels[4]
    df$group <- factor(df$group, levels = levels, ordered = TRUE)
    
    p <- ggplot(df, aes(x = position, y = example, fill = group)) + 
      scale_fill_manual(values = c("#B5EAD7", "#FFDAC1", 
                                   "#FF9AA2", "#FFFFFF"), 
                        labels = levels, drop = FALSE) +
      labs(y = NULL, title = title)
  } else {
    if (sum(toupper(df$group) == "C") == 0) {
      legend_label <- "solid black border (grammar position), no border (background position)"
    } else {
      legend_label <- "solid black border (grammar position), dashed gray border (confounding position), no border (background position)"
    }
    levels <- c("Grammar position",
                "Confounding position",
                "Background position")
    df$group[toupper(df$group) == "G"] <- levels[1]
    df$group[toupper(df$group) == "C"] <- levels[2]
    df$group[toupper(df$group) == "_"] <- levels[3]
    df$group <- factor(df$group, levels = levels, ordered = TRUE)
    
    p <- ggplot(df, aes(x = position, y = example, fill = value,
                        linetype = group, color = group)) + 
      scale_fill_gradient(low = "white", high = muted("blue")) +
      scale_linetype_manual(values = c("solid", "dashed", "blank"), 
                            labels = levels, drop = FALSE, guide = FALSE) +
      scale_color_manual(values = c("black", "gray", "transparent"), 
                         labels = levels, drop = FALSE, guide = FALSE) +
      labs(y = NULL, title = title, caption = legend_label)
  }
  
  p <- p + geom_tile() + 
    scale_x_continuous(breaks = pretty_breaks(n = 5), expand = c(0, 0)) + 
    facet_wrap(vars(label), ncol = 1, scales = "free_y") +
    scale_y_discrete(expand = c(0, 0)) + 
    guides(fill = guide_legend(nrow = 2, byrow = TRUE)) + 
    theme_bw() +
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank(),
          legend.title = element_blank(),
          legend.position = "top",
          legend.key = element_rect(fill = "white", color = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          strip.background = element_blank(),
          plot.margin = margin(t = 10, r = 15, b = 10, l = 10, unit = "pt"))
  
  ggsave(plot = p, filename = output_file_name, width = 7,
         height = 1.5 + length(unique(df$label)) * 1.5)
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 2) {
  plot_agreement(args[1], args[2])
} else if (length(args) == 3) {
  plot_agreement(args[1], args[2], args[3])
} else {
  stop("input file name and output file name must be specified")
}
