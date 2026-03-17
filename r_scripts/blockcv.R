library(sf)
library(blockCV)
library(optparse)

# Arguments
option_list <- list(
  make_option("--points", type = "character"),
  make_option("--output", type = "character"),
  make_option("--folds", type = "integer"),
  make_option("--blocksize", type = "double"),
  make_option("--column", type = "character"),
  make_option("--selection", type = "character"),
  make_option("--iteration", type = "integer"),
  make_option("--seed", type = "integer")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

# Read data
pts <- st_read(opt$points, quiet = TRUE)

# Spatial blocking
cat("Running spatial blocking\n")

sb <- cv_spatial(
  x = pts,
  column = opt$column,
  k = opt$folds,
  size = opt$blocksize,
  selection = opt$selection,
  iteration = opt$iteration,
  biomod2 = FALSE,
  progress = FALSE
)

# Attach fold IDs
pts$fold_id <- sb$folds_ids

# Write output
st_write(pts, opt$output, delete_dsn = TRUE, quiet = TRUE)
