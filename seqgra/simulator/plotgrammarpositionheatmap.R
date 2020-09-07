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

plot_heatmap <- function(input_file_name, output_file_name) {
    # df with label, position, grammar_probability
    label <- position <- grammar_probability <- NULL
    
    df <- read.table(input_file_name, header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE)
    
    df$label <- paste0("label: ", df$label)
    df$label <- factor(df$label,
                       levels = stringr::str_sort(unique(df$label), 
                                                  numeric = TRUE))
    
    p_heatmap <- ggplot(df, aes(x = position, y = 1,
                                fill = grammar_probability)) +  
        geom_tile() + 
        scale_x_continuous(breaks = pretty_breaks(n = 5), expand = c(0, 0)) + 
        scale_fill_gradient(low = "white", high = muted("red")) +
        facet_wrap(vars(label), ncol = 1, scales = "free_y") +
        scale_y_discrete(expand = c(0, 0)) +  
        labs(y = NULL, title = "Grammar position heatmap") +
        theme_bw() +
        theme(axis.text.y = element_blank(),
              axis.ticks.y = element_blank(),
              legend.title = element_blank(),
              legend.position = "none",
              legend.key = element_rect(fill = "white", color = "black"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              strip.background = element_blank(),
              plot.margin = margin(t = 10, r = 15, b = 10, l = 10, unit = "pt"))
    
    p_histogram <- ggplot(df, aes(x = grammar_probability)) +  
        geom_histogram(aes(y = ..density..), color = "black", fill = "white",
                       bins = 30) +
        geom_density(alpha = 0.2, fill = "#66FF66") +
        facet_wrap(vars(label), ncol = 1, scales = "free_y") +
        labs(x = "positional grammar probability", y = NULL, title = "") +
        theme_bw() +
        theme(axis.text.y = element_blank(),
              axis.ticks.y = element_blank(),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              strip.background = element_blank(),
              plot.margin = margin(t = 10, r = 15, b = 10, l = 10, unit = "pt"))
    
    grDevices::pdf(NULL)
    gp1 <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(p_heatmap))
    gp2 <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(p_histogram))
    invisible(grDevices::dev.off())
    
    gp1$heights <- gp2$heights
    p = gridExtra::arrangeGrob(gp1, gp2, ncol = 2, widths = c(2, 1))
    
    ggsave(plot = p, filename = output_file_name, width = 7,
           height = 0.8 + length(unique(df$label)) * 0.6, limitsize = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 2) {
    plot_heatmap(args[1], args[2])
} else {
    stop("input file name and output file name must be specified")
}
