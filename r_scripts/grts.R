library(sf)
library(spsurvey)
library(readr)
library(optparse)

# Arguments
option_list <- list(
  make_option("--points", type="character"),
  make_option("--targets", type="character"),
  make_option("--stratum_var", type="character"),
  make_option("--output", type="character"),
  make_option("--seed", type="integer", default=42)
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

# Read data
points <- st_read(opt$points, quiet = TRUE)
#targets <- read_csv(opt$targets, show_col_types = FALSE)
targets <- read_delim(opt$targets, delim = ";", show_col_types = FALSE)

# Build named vector of target sample sizes
n_base <- setNames(targets$target_n, targets$stratum_id)

# Run GRTS
cat("Running GRTS\n")

grts_result <- try(
  grts(
    sframe = points,
    n_base = n_base,
    stratum_var = opt$stratum_var
  ),
  silent = TRUE
)

if (inherits(grts_result, "try-error")) {
  cat("\n--- GRTS INPUT ERROR DETAILS ---\n")
  stopprnt()
  stop("Stopping due to GRTS input validation error.")
}


# Extract sampled points (CORRECT for spsurvey 5.6.0)
sampled_sf <- grts_result$sites_base

# Write output
st_write(sampled_sf, opt$output, delete_dsn = TRUE, quiet = TRUE)

cat("Output written successfully\n")

