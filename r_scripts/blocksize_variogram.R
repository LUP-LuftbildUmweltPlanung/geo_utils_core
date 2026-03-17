# ============================================================
# Blocksize-Berechnung via Variogramm für BlockCV
# ============================================================

# Pakete
library(tidyverse)
library(sf)
library(gstat)
library(sp)

# ------------------------------------------------------------
# 1. Daten laden
# ------------------------------------------------------------
df_fruehjahr <- read_csv("C:/Users/frede/Downloads/s2_fruehjahr.csv")
df_sommer    <- read_csv("C:/Users/frede/Downloads/s2_sommer.csv")
df_winter    <- read_csv("C:/Users/frede/Downloads/s2_herbst.csv")

# Koordinaten aus .geo-Spalte extrahieren (JSON-String parsen)
parse_coords <- function(df) {
  coords <- df$.geo %>%
    str_extract_all('[-0-9.]+') %>%
    map(as.numeric)

  df %>%
    mutate(
      lon = map_dbl(coords, 1),
      lat = map_dbl(coords, 2)
    )
}

df_fruehjahr <- parse_coords(df_fruehjahr)
df_sommer    <- parse_coords(df_sommer)
df_winter    <- parse_coords(df_winter)

# ------------------------------------------------------------
# 2. Als sf-Objekt (WGS84 → UTM32N für metrische Distanzen!)
# ------------------------------------------------------------
to_utm <- function(df) {
  df %>%
    st_as_sf(coords = c("lon", "lat"), crs = 4326) %>%
    st_transform(crs = 32632)   # UTM Zone 32N (für Deutschland)
}

sf_fruehjahr <- to_utm(df_fruehjahr)
sf_sommer    <- to_utm(df_sommer)
sf_winter    <- to_utm(df_winter)

# ------------------------------------------------------------
# 3. Variogramm berechnen (Beispiel: B8 als Zielvariable)
#    → Wiederhole für alle Bänder und alle Perioden
# ------------------------------------------------------------
fit_variogram <- function(sf_obj, variable = "B8", label = "") {

  # sf → sp (für gstat)
  sp_obj <- as(sf_obj, "Spatial")

  # Empirisches Variogramm
  coords_matrix <- coordinates(sp_obj)

  # Maximale Distanz = halbe Diagonale des Untersuchungsgebiets
  max_dist <- max(dist(coords_matrix)) / 2

  vgm_emp <- variogram(
    object  = as.formula(paste(variable, "~ 1")),
    data    = sp_obj,
    cutoff  = max_dist,
    width   = max_dist / 20   # 20 Distanzklassen
  )

  # Theoretisches Variogramm anpassen (Modell: Spherical)
  vgm_ini <- vgm(
    psill  = var(sp_obj[[variable]], na.rm = TRUE),
    model  = "Sph",
    range  = max_dist / 3,
    nugget = min(vgm_emp$gamma)
  )

  vgm_fit <- fit.variogram(vgm_emp, model = vgm_ini)

  # Range extrahieren
  range_m <- vgm_fit$range[2]

  cat(sprintf("\n--- %s | Variable: %s ---\n", label, variable))
  cat(sprintf("  Nugget : %.1f\n", vgm_fit$psill[1]))
  cat(sprintf("  Sill   : %.1f\n", sum(vgm_fit$psill)))
  cat(sprintf("  Range  : %.0f m  ← Empfohlene Blocksize\n", range_m))

  # Plot
  p <- plot(vgm_emp, vgm_fit,
            main = paste0(label, " | ", variable,
                          " | Range = ", round(range_m), " m"))
  print(p)

  return(list(
    variogram = vgm_emp,
    fit       = vgm_fit,
    range_m   = range_m
  ))
}

# ------------------------------------------------------------
# 4. Alle Bänder und Perioden durchlaufen
# ------------------------------------------------------------
baender  <- c("B2", "B3", "B4", "B8")
perioden <- list(
  Fruehjahr = sf_fruehjahr,
  Sommer    = sf_sommer,
  Winter    = sf_winter
)

ergebnisse <- list()

for (periode in names(perioden)) {
  for (band in baender) {
    key <- paste(periode, band, sep = "_")
    ergebnisse[[key]] <- fit_variogram(
      sf_obj   = perioden[[periode]],
      variable = band,
      label    = periode
    )
  }
}

# ------------------------------------------------------------
# 5. Übersichtstabelle aller Ranges
# ------------------------------------------------------------
range_tabelle <- tibble(
  Periode  = str_extract(names(ergebnisse), "^[^_]+"),
  Band     = str_extract(names(ergebnisse), "[^_]+$"),
  Range_m  = map_dbl(ergebnisse, "range_m")
)

print(range_tabelle)

# Empfehlung: Median über alle Bänder/Perioden als finale Blocksize
empfohlene_blocksize <- median(range_tabelle$Range_m)
cat(sprintf("\n✔ Empfohlene finale Blocksize: %.0f m\n", empfohlene_blocksize))

