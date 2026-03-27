#!/usr/bin/env Rscript
# LUMEN v2 — R Package Installer
# Run once: Rscript scripts/r_bridge/install_packages.R

cat("=== LUMEN v2 R Package Installation ===\n\n")

repo <- "https://cloud.r-project.org"

required <- c("metafor", "meta", "jsonlite", "dplyr")
optional <- c("ggplot2", "forestplot", "gt", "renv")

cat("Installing required packages...\n")
for (pkg in required) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, repos = repo, quiet = TRUE)
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

cat("\nInstalling optional packages...\n")
for (pkg in optional) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    tryCatch(
      install.packages(pkg, repos = repo, quiet = TRUE),
      error = function(e) cat(sprintf("  WARNING: %s failed: %s\n", pkg, e$message))
    )
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

cat("\n=== Verification ===\n")
for (pkg in required) {
  ok <- requireNamespace(pkg, quietly = TRUE)
  status <- if (ok) "OK" else "FAILED"
  ver <- if (ok) as.character(packageVersion(pkg)) else "N/A"
  cat(sprintf("  %s: %s (v%s)\n", pkg, status, ver))
}
cat("\nDone.\n")
