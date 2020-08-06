#!/usr/bin/env Rscript

library(methods)

# hack apparently necessary on Windows if Sys.getenv("HOME") returns
# C:/Users/[User] instead of C:/Users/[User]\Documents
if (.Platform$OS.type == "windows") {
  user_site_path <- Sys.getenv("R_LIBS_USER")
  if (!dir.exists(user_site_path)) {
    warning(paste0("attempting to fix incorrectly set R_LIBS_USER environment variable: ", user_site_path))
    user_site_path <- gsub("//", "/", gsub("\\", "/", user_site_path, fixed = TRUE), fixed = TRUE)
    home_path <- gsub("//", "/", gsub("\\", "/", Sys.getenv("HOME"), fixed = TRUE), fixed = TRUE)
    user_site_path <- gsub(home_path, "", user_site_path)
    user_site_path <- gsub("//", "/", paste0(home_path, "/Documents/", user_site_path), fixed = TRUE)
    .libPaths(c(.libPaths(), user_site_path))
  }
}

if (!("ggplot2" %in% rownames(installed.packages())) ||
    !("scales" %in% rownames(installed.packages()))) {
  stop(paste0("R packages ggplot2 and/or scales are missing (lib paths: ",
    paste(.libPaths(), collapse = "; "), ")"))
}

library(ggplot2)
library(scales)

plot_agreement <- function(input_file_name, output_file_name, title,
                           caption = NULL) {
  # df with example, position, group, label, precision, recall, 
  # specificity, f1, n
  # where precision, recall, specificity, f1 are mean values per label,
  # and n is the number of examples per label
  # if df contains value column, switch to non-thresholded mode
  example <- position <- label <- group <- NULL

  if (!is.null(title) && title == "") {
    title <- NULL
  }
  if (!is.null(caption) && caption == "") {
    caption <- NULL
  }

  if (!is.null(caption)) {
    caption <- gsub(":NL:", "\n", caption, fixed = TRUE)
  }
  
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
  df$label <- factor(df$label,
                     levels = stringr::str_sort(unique(df$label), 
                                                numeric = TRUE))
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
      guides(fill = guide_legend(nrow = 2, byrow = TRUE))
  } else {
    df$value <- df$value * ifelse(df$group == "G", 1, -1)
    levels <- c("Grammar position",
                "Confounding position",
                "Background position")
    df$group[toupper(df$group) == "G"] <- levels[1]
    df$group[toupper(df$group) == "C"] <- levels[2]
    df$group[toupper(df$group) == "_"] <- levels[3]
    df$group <- factor(df$group, levels = levels, ordered = TRUE)
    

    p <- ggplot(df, aes(x = position, y = example, fill = value)) + 
      scale_fill_gradient2(low = muted("red"), mid = "white",
                           high = muted("green"), midpoint = 0,
                           limits = c(-1, 1), guide = FALSE)
  }
  
  p <- p + geom_tile() + 
    scale_x_continuous(breaks = pretty_breaks(n = 5), expand = c(0, 0)) + 
    facet_wrap(vars(label), ncol = 1, scales = "free_y") +
    scale_y_discrete(expand = c(0, 0)) +  
    labs(y = NULL, title = title, caption = caption) +
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

if (length(args) == 3) {
  plot_agreement(args[1], args[2], args[3])
} else if (length(args) == 4) {
  plot_agreement(args[1], args[2], args[3], args[4])
} else {
  stop("input file name, output file name, and plot title must be specified")
}
