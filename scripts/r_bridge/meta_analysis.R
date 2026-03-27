#!/usr/bin/env Rscript
# LUMEN v2 — R metafor Bridge
# Reads JSON input, runs meta-analysis via metafor, outputs JSON results.
#
# Usage:
#   Rscript meta_analysis.R --input data.json --output results.json
#   Rscript meta_analysis.R --input data.json --output results.json --config config.json
#
# Input JSON schema:
# {
#   "effects": [0.5, 0.3, ...],        # effect sizes (e.g., Hedges' g, log-RR)
#   "variances": [0.1, 0.05, ...],     # sampling variances
#   "labels": ["Study A", ...],        # study labels
#   "years": [2020, 2021, ...],        # optional: publication years
#   "subgroups": ["A", "B", ...],      # optional: subgroup variable
#   "moderators": [1.5, 2.0, ...],     # optional: continuous moderator
#   "method": "REML",                  # estimation method
#   "knha": true,                      # Hartung-Knapp adjustment
#   "measure_label": "SMD"             # effect size label
# }

# --- Setup ---
.libPaths(c(Sys.getenv("R_LIBS_USER", "~/R/library"), .libPaths()))
suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
input_file <- NULL
output_file <- NULL
config_file <- NULL
figures_dir <- NULL

i <- 1
while (i <= length(args)) {
  if (args[i] == "--input" && i < length(args)) {
    input_file <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--output" && i < length(args)) {
    output_file <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--config" && i < length(args)) {
    config_file <- args[i + 1]; i <- i + 2
  } else if (args[i] == "--figures-dir" && i < length(args)) {
    figures_dir <- args[i + 1]; i <- i + 2
  } else {
    i <- i + 1
  }
}

if (is.null(input_file) || is.null(output_file)) {
  stop("Usage: Rscript meta_analysis.R --input data.json --output results.json")
}

# --- Load data ---
data <- fromJSON(input_file)
yi <- as.numeric(data$effects)
vi <- as.numeric(data$variances)
slab <- if (!is.null(data$labels)) data$labels else paste0("Study_", seq_along(yi))
method <- if (!is.null(data$method)) data$method else "REML"
knha <- if (!is.null(data$knha)) data$knha else TRUE
measure_label <- if (!is.null(data$measure_label)) data$measure_label else "SMD"

k <- length(yi)
results <- list(
  engine = "r_metafor",
  metafor_version = as.character(packageVersion("metafor")),
  k = k,
  measure = measure_label
)

# --- Helper: safe numeric extraction ---
safe_num <- function(x, default = NA) {
  val <- tryCatch(as.numeric(x), error = function(e) default)
  if (length(val) == 0 || is.null(val)) default else val[1]
}

# ====================================================================
# 1. MAIN META-ANALYSIS
# ====================================================================
tryCatch({
  res <- rma(yi = yi, vi = vi, method = method, slab = slab,
             test = if (knha) "knha" else "z")

  results$main <- list(
    pooled_effect = safe_num(res$beta),
    se = safe_num(res$se),
    ci_lower = safe_num(res$ci.lb),
    ci_upper = safe_num(res$ci.ub),
    z_value = safe_num(res$zval),
    p_value = safe_num(res$pval),
    tau2 = safe_num(res$tau2),
    tau2_se = safe_num(res$se.tau2),
    I2 = safe_num(res$I2),
    H2 = safe_num(res$H2),
    Q = safe_num(res$QE),
    Q_df = k - 1,
    Q_p = safe_num(res$QEp),
    method = method,
    knha = knha,
    k = k
  )

  # Prediction interval (if k >= 3)
  if (k >= 3) {
    pi_res <- tryCatch({
      pred <- predict(res)
      list(pi_lower = safe_num(pred$pi.lb), pi_upper = safe_num(pred$pi.ub))
    }, error = function(e) list(pi_lower = NA, pi_upper = NA))
    results$main <- c(results$main, pi_res)
  }

  # Per-study weights
  w <- weights(res)
  results$study_weights <- as.list(setNames(round(as.numeric(w), 4), slab))

}, error = function(e) {
  results$main <<- list(error = paste("Main analysis failed:", e$message))
  cat(sprintf("WARNING: Main analysis failed: %s\n", e$message), file = stderr())
})

# ====================================================================
# 2. SENSITIVITY: LEAVE-ONE-OUT
# ====================================================================
tryCatch({
  if (k >= 3) {
    loo <- leave1out(res)
    loo_list <- list()
    for (j in seq_len(length(loo$estimate))) {
      loo_list[[j]] <- list(
        excluded = slab[j],
        estimate = safe_num(loo$estimate[j]),
        se = safe_num(loo$se[j]),
        ci_lower = safe_num(loo$ci.lb[j]),
        ci_upper = safe_num(loo$ci.ub[j]),
        p_value = safe_num(loo$pval[j]),
        tau2 = safe_num(loo$tau2[j]),
        I2 = safe_num(loo$I2[j]),
        Q = safe_num(loo$Q[j])
      )
    }
    results$leave_one_out <- loo_list
  }
}, error = function(e) {
  results$leave_one_out <<- list(error = e$message)
})

# ====================================================================
# 3. SENSITIVITY: CUMULATIVE META-ANALYSIS
# ====================================================================
tryCatch({
  if (k >= 3 && !is.null(data$years)) {
    years <- as.numeric(data$years)
    ord <- order(years)
    res_cum <- rma(yi = yi[ord], vi = vi[ord], method = method,
                   slab = slab[ord], test = if (knha) "knha" else "z")
    cum <- cumul(res_cum)
    cum_list <- list()
    for (j in seq_len(length(cum$estimate))) {
      cum_list[[j]] <- list(
        added_study = slab[ord[j]],
        year = years[ord[j]],
        estimate = safe_num(cum$estimate[j]),
        ci_lower = safe_num(cum$ci.lb[j]),
        ci_upper = safe_num(cum$ci.ub[j]),
        tau2 = safe_num(cum$tau2[j])
      )
    }
    results$cumulative <- cum_list
  }
}, error = function(e) {
  results$cumulative <<- list(error = e$message)
})

# ====================================================================
# 4. PUBLICATION BIAS
# ====================================================================
# 4a. Egger's regression test
tryCatch({
  if (k >= 3) {
    eg <- regtest(res, model = "lm")
    results$egger_test <- list(
      intercept = safe_num(eg$est),
      se = safe_num(eg$se),
      z_value = safe_num(eg$zval),
      p_value = safe_num(eg$pval),
      significant = safe_num(eg$pval) < 0.10
    )
  }
}, error = function(e) {
  results$egger_test <<- list(error = e$message)
})

# 4b. Rank correlation (Begg's test)
tryCatch({
  if (k >= 3) {
    rk <- ranktest(res)
    results$begg_test <- list(
      tau = safe_num(rk$tau),
      p_value = safe_num(rk$pval),
      significant = safe_num(rk$pval) < 0.10
    )
  }
}, error = function(e) {
  results$begg_test <<- list(error = e$message)
})

# 4c. Trim-and-fill
tryCatch({
  if (k >= 5) {
    tf <- trimfill(res)
    results$trim_and_fill <- list(
      k_original = k,
      k_filled = safe_num(tf$k0),
      k_total = k + safe_num(tf$k0),
      adjusted_estimate = safe_num(tf$beta),
      adjusted_ci_lower = safe_num(tf$ci.lb),
      adjusted_ci_upper = safe_num(tf$ci.ub),
      adjusted_p_value = safe_num(tf$pval),
      side = as.character(tf$side)
    )
  }
}, error = function(e) {
  results$trim_and_fill <<- list(error = e$message)
})

# 4d. Failsafe N (Rosenthal)
tryCatch({
  if (k >= 2) {
    fsn_res <- fsn(x = res, type = "Rosenthal")
    results$failsafe_n <- list(
      fsn = safe_num(fsn_res$fsnum),
      p_value = safe_num(fsn_res$pval),
      target_alpha = 0.05
    )
  }
}, error = function(e) {
  results$failsafe_n <<- list(error = e$message)
})

# ====================================================================
# 5. SUBGROUP ANALYSIS
# ====================================================================
tryCatch({
  if (!is.null(data$subgroups)) {
    sg <- as.character(data$subgroups)
    groups <- unique(sg)
    if (length(groups) >= 2 && length(groups) <= 10) {
      # Overall test for subgroup differences
      res_mod <- rma(yi = yi, vi = vi, mods = ~ factor(sg), method = method,
                     test = if (knha) "knha" else "z")
      results$subgroup_analysis <- list(
        variable = "subgroup",
        Q_between = safe_num(res_mod$QM),
        Q_between_df = length(groups) - 1,
        Q_between_p = safe_num(res_mod$QMp),
        groups = list()
      )

      # Per-group analysis
      for (g in groups) {
        idx <- which(sg == g)
        if (length(idx) >= 2) {
          res_g <- tryCatch(
            rma(yi = yi[idx], vi = vi[idx], method = method,
                slab = slab[idx], test = if (knha) "knha" else "z"),
            error = function(e) NULL
          )
          if (!is.null(res_g)) {
            results$subgroup_analysis$groups[[g]] <- list(
              k = length(idx),
              pooled_effect = safe_num(res_g$beta),
              ci_lower = safe_num(res_g$ci.lb),
              ci_upper = safe_num(res_g$ci.ub),
              p_value = safe_num(res_g$pval),
              tau2 = safe_num(res_g$tau2),
              I2 = safe_num(res_g$I2)
            )
          }
        } else {
          results$subgroup_analysis$groups[[g]] <- list(
            k = length(idx),
            note = "Insufficient studies for pooling"
          )
        }
      }
    }
  }
}, error = function(e) {
  results$subgroup_analysis <<- list(error = e$message)
})

# ====================================================================
# 6. META-REGRESSION
# ====================================================================
tryCatch({
  if (!is.null(data$moderators)) {
    mod_vals <- as.numeric(data$moderators)
    if (sum(!is.na(mod_vals)) >= 3) {
      res_reg <- rma(yi = yi, vi = vi, mods = ~ mod_vals, method = method,
                     test = if (knha) "knha" else "z")
      results$meta_regression <- list(
        moderator = "moderator",
        intercept = safe_num(res_reg$beta[1]),
        slope = safe_num(res_reg$beta[2]),
        slope_se = safe_num(res_reg$se[2]),
        slope_p = safe_num(res_reg$pval[2]),
        QM = safe_num(res_reg$QM),
        QM_p = safe_num(res_reg$QMp),
        QE = safe_num(res_reg$QE),
        QE_p = safe_num(res_reg$QEp),
        R2 = safe_num(res_reg$R2),
        tau2_residual = safe_num(res_reg$tau2)
      )
    }
  }
}, error = function(e) {
  results$meta_regression <<- list(error = e$message)
})

# ====================================================================
# 7. INFLUENCE DIAGNOSTICS
# ====================================================================
tryCatch({
  if (k >= 3) {
    inf <- influence(res)
    inf_list <- list()
    for (j in seq_len(k)) {
      inf_list[[j]] <- list(
        study = slab[j],
        hat = safe_num(inf$inf$hat[j]),
        cooks_distance = safe_num(inf$inf$cook.d[j]),
        dfbetas = safe_num(inf$inf$dfbs[j]),
        dffits = safe_num(inf$inf$dffits[j]),
        covratio = safe_num(inf$inf$cov.r[j]),
        tau2_without = safe_num(inf$inf$tau2.del[j]),
        weight = safe_num(inf$inf$weight[j])
      )
    }
    results$influence <- inf_list
  }
}, error = function(e) {
  results$influence <<- list(error = e$message)
})

# ====================================================================
# 8. PUBLICATION-QUALITY FIGURES (300 DPI)
# ====================================================================
if (!is.null(figures_dir) && exists("res") && !is.null(res)) {
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  has_meta <- requireNamespace("meta", quietly = TRUE)
  has_ggplot <- requireNamespace("ggplot2", quietly = TRUE)

  tryCatch({
    # 8a. Forest plot via metafor (300 DPI)
    png(file.path(figures_dir, "forest_plot_r.png"),
        width = 10, height = max(4, 0.4 * k + 2), units = "in", res = 300)
    forest(res, header = TRUE, xlab = measure_label,
           mlab = sprintf("RE Model (REML, %s)", if (knha) "HKSJ" else "z"),
           addpred = TRUE, col = "darkblue",
           fonts = "sans", cex = 0.85)
    dev.off()
    cat("  Forest plot saved\n", file = stderr())
  }, error = function(e) {
    cat(sprintf("  WARNING: Forest plot failed: %s\n", e$message), file = stderr())
  })

  tryCatch({
    # 8b. Funnel plot (300 DPI)
    png(file.path(figures_dir, "funnel_plot_r.png"),
        width = 8, height = 6, units = "in", res = 300)
    funnel(res, main = "Funnel Plot", xlab = measure_label,
           back = "white", shade = c("white", "gray90"),
           hlines = "gray80", legend = TRUE)
    dev.off()
    cat("  Funnel plot saved\n", file = stderr())
  }, error = function(e) {
    cat(sprintf("  WARNING: Funnel plot failed: %s\n", e$message), file = stderr())
  })

  tryCatch({
    # 8c. Baujat plot (300 DPI)
    if (k >= 3) {
      png(file.path(figures_dir, "baujat_plot_r.png"),
          width = 8, height = 6, units = "in", res = 300)
      baujat(res, main = "Baujat Plot")
      dev.off()
      cat("  Baujat plot saved\n", file = stderr())
    }
  }, error = function(e) {
    cat(sprintf("  WARNING: Baujat plot failed: %s\n", e$message), file = stderr())
  })

  tryCatch({
    # 8d. Trim-and-fill funnel (300 DPI)
    if (k >= 5) {
      tf_res <- trimfill(res)
      png(file.path(figures_dir, "trimfill_funnel_r.png"),
          width = 8, height = 6, units = "in", res = 300)
      funnel(tf_res, main = "Trim-and-Fill Funnel Plot", xlab = measure_label,
             legend = TRUE)
      dev.off()
      cat("  Trim-and-fill funnel saved\n", file = stderr())
    }
  }, error = function(e) {
    cat(sprintf("  WARNING: Trim-fill plot failed: %s\n", e$message), file = stderr())
  })

  # 8e. Summary table via gt (if available)
  if (requireNamespace("gt", quietly = TRUE)) {
    tryCatch({
      library(gt)
      tbl_data <- data.frame(
        Study = slab,
        Effect = round(yi, 4),
        SE = round(sqrt(vi), 4),
        CI_Lower = round(yi - 1.96 * sqrt(vi), 4),
        CI_Upper = round(yi + 1.96 * sqrt(vi), 4),
        Weight = round(as.numeric(weights(res)), 2)
      )
      # Append pooled row
      pooled_row <- data.frame(
        Study = "Pooled (RE)",
        Effect = round(safe_num(res$beta), 4),
        SE = round(safe_num(res$se), 4),
        CI_Lower = round(safe_num(res$ci.lb), 4),
        CI_Upper = round(safe_num(res$ci.ub), 4),
        Weight = 100.0
      )
      tbl_data <- rbind(tbl_data, pooled_row)

      gt_tbl <- gt(tbl_data) |>
        tab_header(title = "Meta-Analysis Summary",
                   subtitle = sprintf("k=%d, Method=%s, %s", k, method,
                                      if (knha) "HKSJ adjustment" else "z-test")) |>
        fmt_number(columns = c(Effect, SE, CI_Lower, CI_Upper), decimals = 4) |>
        fmt_number(columns = Weight, decimals = 2) |>
        tab_style(style = cell_text(weight = "bold"),
                  locations = cells_body(rows = Study == "Pooled (RE)"))

      gtsave(gt_tbl, file.path(figures_dir, "summary_table.html"))
      cat("  Summary table (HTML) saved\n", file = stderr())
    }, error = function(e) {
      cat(sprintf("  WARNING: gt table failed: %s\n", e$message), file = stderr())
    })
  }

  results$figures_generated <- list(
    directory = figures_dir,
    dpi = 300,
    files = list.files(figures_dir, pattern = "\\.(png|html)$")
  )
}

# ====================================================================
# 9. AUDIT TRAIL
# ====================================================================
results$audit <- list(
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  metafor_version = as.character(packageVersion("metafor")),
  input_file = normalizePath(input_file, mustWork = FALSE),
  output_file = normalizePath(output_file, mustWork = FALSE),
  input_hash = tryCatch(
    as.character(tools::md5sum(input_file)),
    error = function(e) "unavailable"
  )
)

# ====================================================================
# OUTPUT
# ====================================================================
json_out <- toJSON(results, auto_unbox = TRUE, pretty = TRUE, na = "null",
                   digits = 8)
writeLines(json_out, output_file)
cat(sprintf("Results written to %s\n", output_file), file = stderr())
cat(sprintf("  k=%d, method=%s, pooled=%.4f [%.4f, %.4f], I2=%.1f%%, tau2=%.4f\n",
            k, method,
            safe_num(results$main$pooled_effect),
            safe_num(results$main$ci_lower),
            safe_num(results$main$ci_upper),
            safe_num(results$main$I2),
            safe_num(results$main$tau2)),
    file = stderr())
