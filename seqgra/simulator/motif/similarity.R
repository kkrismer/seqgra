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
    !("stringr" %in% rownames(installed.packages())) ||
    !("dplyr" %in% rownames(installed.packages()))) {
    stop(paste0("R packages ggplot2, stringr and/or dplyr are missing (lib paths: ",
                paste(.libPaths(), collapse = "; "), ")"))
}

library(ggplot2)

create_similarity_matrix_plot <- function(grammar_folder,
                                          similarity_measure = c("ess", 
                                                                 "kld")) {
    se1 <- se2 <- score <- NULL
    similarity_measure <- match.arg(similarity_measure, c("ess", "kld"))
                              
    input_file <- paste0(grammar_folder, "/motif-", similarity_measure, 
                          "-matrix.txt")
    output_file <- paste0(grammar_folder, "/motif-", similarity_measure, 
                          "-matrix.pdf")
    if (file.exists(input_file)) {
        df <- read.table(input_file, header = TRUE, sep = "\t",
                         stringsAsFactors = FALSE)
        colnames(df) <- c("se1", "se2", "score")
        
        if (similarity_measure == "ess") {
            df <- dplyr::group_by(df, se1)
            min_df <- dplyr::summarize(df, min(score), .groups = "drop")
            colnames(min_df) <- c("se1", "min_score")
            min_df$min_score <- pmin(min_df$min_score, 0)
            df <- dplyr::inner_join(df, min_df, by = "se1")
            df$score <- df$score + abs(df$min_score)
            df$min_score <- NULL
            
            self_df <- dplyr::filter(df, se1 == se2)
            self_df$se2 <- NULL
            colnames(self_df) <- c("se1", "self_score")
            df <- dplyr::inner_join(df, self_df, by = "se1")
            df$relative_score <- df$score / df$self_score
            
            df$score_label <- paste0(round(df$relative_score * 100), "%")
        } else {
            df$relative_score <- df$score
            df$score_label <- round(df$relative_score, 2)
            temp <- df$se1
            df$se1 <- df$se2
            df$se2 <- temp
        }
        
        df$se1 <- factor(df$se1, levels = stringr::str_sort(unique(df$se1),
                                                            numeric = TRUE))
        df$se2 <- factor(df$se2, levels = stringr::str_sort(unique(df$se2),
                                                            numeric = TRUE))
        p <- ggplot(df, aes(x = se1, y = se2, fill = relative_score)) +
            geom_tile(color = "white") +
            coord_fixed() +
            labs(x = NULL, y = NULL) +
            theme_minimal() +
            theme(
                plot.title.position = "plot",
                axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
                panel.grid.major = element_blank(),
                panel.border = element_blank(),
                panel.background = element_blank(),
                axis.ticks = element_blank(),
                legend.position = "top")
        
        if (similarity_measure == "ess") {
            p <- p + scale_fill_gradient(low = "blue", high = "red",
                                         space = "Lab",
                                         name = "empirical similarity score")
            
        } else {
            p <- p + scale_fill_gradient(low = "red", high = "blue",
                                         space = "Lab",
                                         name = "KL divergence")
        }
        
        if (length(unique(df$se1)) < 30) {
            p <- p + geom_text(aes(x = se1, y = se2, label = score_label),
                               size = 1.5)
        }
        ggsave(plot = p, filename = output_file,
               width = length(unique(df$se1)) * 0.2 + 2,
               height = length(unique(df$se2)) * 0.2 + 2, limitsize = FALSE)
    }
}

create_violin_plot <- function(df, out_file,
                               similarity_measure = c("ess", "kld")) {
    similarity_measure <- match.arg(similarity_measure, c("ess", "kld"))
    similarity_measure_label <- ifelse(similarity_measure == "ess",
                                       "empirical similarity score",
                                       "KL divergence") 
    temp_df <- dplyr::filter(df, is_mean)
    
    if (similarity_measure == "ess") {
        temp_df <- dplyr::arrange(temp_df, score)
    } else {
        temp_df <- dplyr::arrange(temp_df, dplyr::desc(score))
    }
    df$se <- factor(df$se, temp_df$se)
    p <- ggplot(df, aes(x = se, y = score)) + 
        geom_violin() +
        geom_point(aes(color = is_mean), alpha = 0.6) +
        labs(x = "sequence element", y = similarity_measure_label, 
             color = NULL) +
        theme_classic() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
              legend.position = "none")
    
    ggsave(plot = p, filename = out_file, 
           width = length(unique(df$se)) * 0.2 + 2, height = 5, 
           limitsize = FALSE)
}

identify_ambiguous_motifs <- function(grammar_folder, 
                                      similarity_measure = c("ess", "kld")) {
    se1 <- se2 <- score <- NULL
    similarity_measure <- match.arg(similarity_measure, c("ess", "kld"))
                              
    input_file <- paste0(grammar_folder, "/motif-", similarity_measure, 
                          "-matrix.txt")
    if (file.exists(input_file)) {
        df <- read.table(input_file, header = TRUE, sep = "\t",
                         stringsAsFactors = FALSE)
        colnames(df) <- c("se1", "se2", "score")
        
        if (similarity_measure == "ess") {
            df <- dplyr::group_by(df, se1)
            min_df <- dplyr::summarize(df, min(score), .groups = "drop")
            colnames(min_df) <- c("se1", "min_score")
            min_df$min_score <- pmin(min_df$min_score, 0)
            df <- dplyr::inner_join(df, min_df, by = "se1")
            df$score <- df$score + abs(df$min_score)
            df$min_score <- NULL
            
            self_df <- dplyr::filter(df, se1 == se2)
            self_df$se2 <- NULL
            colnames(self_df) <- c("se1", "self_score")
            df <- dplyr::inner_join(df, self_df, by = "se1")
            df$score <- df$score / df$self_score
        }
        df <- dplyr::filter(df, se1 != se2)
        
        # group by se1
        mean_df <- dplyr::group_by(df, se1)
        mean_df <- dplyr::summarize(mean_df, score = mean(score), 
                                    .groups = "drop")
        mean_df$se2 <- "mean"
        se1_df <- dplyr::bind_rows(df, mean_df)
        se1_df$is_mean <- se1_df$se2 == "mean"
        se1_df$se <- se1_df$se1
        se1_df$se1 <- NULL
        se1_df$se2 <- NULL
        out_file <- paste0(grammar_folder, "/motif-",
                           similarity_measure, "-se1-violin.pdf")
        create_violin_plot(se1_df, out_file, similarity_measure)
        
        # group by se2
        mean_df <- dplyr::group_by(df, se2)
        mean_df <- dplyr::summarize(mean_df, score = mean(score), 
                                    .groups = "drop")
        mean_df$se1 <- "mean"
        se2_df <- dplyr::bind_rows(df, mean_df)
        se2_df$is_mean <- se2_df$se1 == "mean"
        se2_df$se <- se2_df$se2
        se2_df$se1 <- NULL
        se2_df$se2 <- NULL
        out_file <- paste0(grammar_folder, "/motif-",
                           similarity_measure, "-se2-violin.pdf")
        create_violin_plot(se2_df, out_file, similarity_measure)

        if (similarity_measure == "ess") {
            df <- dplyr::filter(se1_df, !is_mean)
            df <- dplyr::group_by(df, se)
            df <- dplyr::summarize(df, mean_score = mean(score),
                                   min_score = min(score),
                                   max_score = max(score),
                                   median_score = median(score), 
                                   .groups = "drop")
            df <- dplyr::arrange(df, mean_score)
        } else {
            df <- dplyr::filter(se2_df, !is_mean)
            df <- dplyr::group_by(df, se)
            df <- dplyr::summarize(df, mean_score = mean(score),
                                   min_score = min(score),
                                   max_score = max(score),
                                   median_score = median(score), 
                                   .groups = "drop")
            df <- dplyr::arrange(df, dplyr::desc(mean_score))
        }
        df <- dplyr::select(df, se, mean_score, min_score, max_score, 
                            median_score)
        out_file <- paste0(grammar_folder, "/motif-", similarity_measure, 
                           "-statistics.txt")
        write.table(df, out_file, quote = FALSE, sep = "\t", row.names = FALSE)
    }
}

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 2) {
    grammar_folder <- args[1]
    similarity_measure <- args[2]
    create_similarity_matrix_plot(grammar_folder, similarity_measure)
    identify_ambiguous_motifs(grammar_folder, similarity_measure)
} else {
    stop("grammar path and similarity score measure must be specified")
}
