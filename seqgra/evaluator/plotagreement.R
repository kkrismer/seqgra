#!/usr/bin/env Rscript

plot_agreement <- function(input_file_name, output_file_name, title = NULL) {
    # df with example, position, group, label, precision, recall, 
    # specificity, f1, n
    # where precision, recall, specificity, f1 are mean values per label,
    # and n is the number of examples per label
    example <- position <- label <- group <- NULL

    df <- read.table(input_file_name, header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE)
    
    levels <- c("TP (grammar position, model position)",
                "FN (grammar position, no model position)",
                "FP (background position, model position)",
                "TN (background position, no model position)")
    df$group[toupper(df$group) == "TP"] <- levels[1]
    df$group[toupper(df$group) == "FN"] <- levels[2]
    df$group[toupper(df$group) == "FP"] <- levels[3]
    df$group[toupper(df$group) == "TN"] <- levels[4]

    df$group <- factor(df$group,
                       levels = levels, ordered = TRUE)

    df$label <- paste0("label: ", df$label,
                       " | precision = ", round(df$precision, digits = 3),
                       ", recall (sensitivity) = ",
                       round(df$recall, digits = 3),
                       ", specificity = ", round(df$specificity, digits = 3), 
                       ", F1 = ", round(df$f1, digits = 3),
                       ", n = ", df$n)
    df$example <- as.factor(df$example)
    p <- ggplot2::ggplot(df, ggplot2::aes(x = position, y = example, 
                                          fill = group)) + 
        ggplot2::geom_tile() + 
        ggplot2::scale_x_continuous(breaks = scales::pretty_breaks(n = 5), 
                                    expand = c(0, 0)) + 
        ggplot2::scale_y_discrete(expand = c(0, 0)) + 
        ggplot2::scale_fill_manual(values = c("#B5EAD7", "#FFDAC1", 
                                              "#FF9AA2", "#FFFFFF"), 
                                   labels = levels,
                                   drop = FALSE) +
        ggplot2::facet_wrap(ggplot2::vars(label), ncol = 1, 
                            scales = "free_y") +
        ggplot2::guides(fill = ggplot2::guide_legend(nrow = 2, byrow = TRUE)) + 
        ggplot2::theme_bw() +
        ggplot2::theme(axis.text.y = ggplot2::element_blank(),
                       axis.ticks.y = ggplot2::element_blank(),
                       legend.title = ggplot2::element_blank(),
                       legend.position = "top",
                       legend.key = ggplot2::element_rect(fill = "white", 
                                                          color = "black"),
                       panel.grid.major = ggplot2::element_blank(),
                       panel.grid.minor = ggplot2::element_blank(),
                       strip.background = ggplot2::element_blank(),
                       plot.margin = ggplot2::margin(t = 10, r = 15, b = 10, 
                                                     l = 10, unit = "pt")) +
        ggplot2::labs(y = NULL, title = title)
    ggplot2::ggsave(plot = p, filename = output_file_name, width = 7,
                    height = 1.5 + length(unique(df$label)) * 1.2)
}

args <- commandArgs(trailingOnly=TRUE)

if (length(args) == 2) {
    plot_agreement(args[1], args[2])
} else if (length(args) == 3) {
    plot_agreement(args[1], args[2], args[3])
} else {
    stop("input file name and output file name must be specified")
}
