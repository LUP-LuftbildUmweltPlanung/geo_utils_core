# ============================================================
# Moran's I Decay – mit Plots pro Band (alle Perioden)
# ============================================================

library(tidyverse)
library(sf)
library(spdep)

# ------------------------------------------------------------
# 1. Daten laden (unverändert)
# ------------------------------------------------------------
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

to_utm <- function(df) {
  df %>%
    st_as_sf(coords = c("lon", "lat"), crs = 4326) %>%
    st_transform(crs = 32632)
}

sf_fruehjahr <- read_csv("C:/Users/frede/Downloads/s2_fruehjahr.csv") %>% parse_coords() %>% to_utm()
sf_sommer    <- read_csv("C:/Users/frede/Downloads/s2_sommer.csv")    %>% parse_coords() %>% to_utm()
sf_herbst    <- read_csv("C:/Users/frede/Downloads/s2_herbst.csv")    %>% parse_coords() %>% to_utm()

perioden <- list(
  Frühjahr = sf_fruehjahr,
  Sommer   = sf_sommer,
  Herbst   = sf_herbst
)

baender <- c("B2", "B3", "B4", "B8")
breaks  <- seq(0, 150000, by = 5000)

# ------------------------------------------------------------
# 2. Moran's I Decay für alle Perioden & Bänder berechnen
#    → Ergebnis: eine lange Tabelle mit allen Decay-Kurven
# ------------------------------------------------------------
decay_all <- map_dfr(names(perioden), function(periode) {
  map_dfr(baender, function(band) {

    coords <- st_coordinates(perioden[[periode]])

    map_dfr(seq_along(breaks[-1]), function(i) {
      nb <- dnearneigh(coords, d1 = breaks[i], d2 = breaks[i+1])
      if (all(card(nb) == 0)) return(NULL)
      w  <- nb2listw(nb, style = "W", zero.policy = TRUE)
      mi <- moran.test(perioden[[periode]][[band]], w, zero.policy = TRUE)
      tibble(
        Periode = periode,
        Band    = band,
        dist_km = breaks[i+1] / 1000,
        moran_I = mi$estimate["Moran I statistic"],
        p_value = mi$p.value
      )
    })
  })
})

# ------------------------------------------------------------
# 3. Schwellenwerte berechnen (I ≤ 0.05)
# ------------------------------------------------------------
thresholds <- decay_all %>%
  group_by(Periode, Band) %>%
  filter(moran_I <= 0.05) %>%
  slice(1) %>%
  ungroup()

# Übersichtstabelle
ergebnisse_moran <- decay_all %>%
  group_by(Periode, Band) %>%
  filter(moran_I <= 0.05) %>%
  slice(1) %>%
  transmute(Threshold_km = dist_km) %>%
  ungroup()

print(ergebnisse_moran)
cat("\nEmpfohlene Blocksize (Median):",
    median(ergebnisse_moran$Threshold_km, na.rm = TRUE), "km\n")

# ------------------------------------------------------------
# 4. Plot pro Band – alle drei Perioden als Linien
# ------------------------------------------------------------
plots <- map(baender, function(band) {

  decay_band     <- decay_all   %>% filter(Band == band)
  threshold_band <- thresholds  %>% filter(Band == band)

  ggplot(decay_band, aes(x = dist_km, y = moran_I,
                         color = Periode, linetype = Periode)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.5) +

    # Schwellenwert-Linien pro Periode (vertikale Linien)
    geom_vline(
      data        = threshold_band,
      aes(xintercept = dist_km, color = Periode),
      linetype    = "dashed",
      linewidth   = 0.7,
      show.legend = FALSE
    ) +

    # Horizontale Referenzlinie I = 0.05
    geom_hline(yintercept = 0.05, color = "black",
               linetype = "dotted", linewidth = 0.7) +

    # Beschriftung der Schwellenwerte
    geom_text(
      data  = threshold_band,
      aes(x = dist_km + 2, y = 0.92, label = paste0(dist_km, " km"),
          color = Periode),
      size        = 3.2,
      hjust       = 0,
      show.legend = FALSE
    ) +

    scale_y_continuous(limits = c(NA, 1)) +   # Y-Achse startet beim Datumminimum, endet bei 1
    scale_color_manual(values = c(
      "Frühjahr" = "steelblue",
      "Sommer"   = "darkorange",
      "Herbst"   = "forestgreen"
    )) +

    labs(
      title    = paste0("Moran's I Decay – Band ", band),
      subtitle = "Gestrichelte Linien = Schwellenwert I ≤ 0.05 je Periode",
      x        = "Distanz (km)",
      y        = "Moran's I",
      color    = "Periode",
      linetype = "Periode"
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
})

# Plots anzeigen
walk(plots, print)

# Optional: alle 4 Plots in einer Abbildung
# library(patchwork)
# (plots[[1]] | plots[[2]]) / (plots[[3]] | plots[[4]])