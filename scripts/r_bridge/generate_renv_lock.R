#!/usr/bin/env Rscript
# LUMEN v2 — Generate renv.lock for reproducibility
#
# Usage:
#   Rscript scripts/r_bridge/generate_renv_lock.R [output_path]
#
# Captures exact package versions used by the R statistical engine.

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1) args[1] else "renv.lock"

cat("=== Generating renv.lock ===\n")

pkgs <- c("metafor", "meta", "jsonlite", "dplyr", "ggplot2", "gt")

lock <- list(
  R = list(Version = paste(R.version$major, R.version$minor, sep = ".")),
  Packages = list()
)

for (pkg in pkgs) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    desc <- packageDescription(pkg)
    lock$Packages[[pkg]] <- list(
      Package = pkg,
      Version = as.character(packageVersion(pkg)),
      Source = if (!is.null(desc$Repository)) desc$Repository else "unknown"
    )
  }
}

json_out <- jsonlite::toJSON(lock, auto_unbox = TRUE, pretty = TRUE)
writeLines(json_out, output_path)
cat(sprintf("renv.lock written to %s (%d packages)\n", output_path, length(lock$Packages)))
