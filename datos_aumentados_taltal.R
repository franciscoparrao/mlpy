#####################
###MAPAS TEMÁTICOS###
#####################

library(terra)
library(Rsagacmd)
library(raster)
library(whitebox)
library(rgee)
library(sf)
library(sp)
library(gridExtra)
library(maptools)
library(data.table)
library(mlr3verse)
library(ggplot2)
library(dplyr)
library(stats)
library(PMCMRplus)
library(rstatix)
library(pracma)
library(data.table)
library(corrplot)
library(tidyr)
library(xtable)

# Preparación básica de los archivos

## Definir la proyección UTM
crs_utm <- "+proj=utm +init=epsg:32619"

## Cargar el DEM
#fabdem_1 <- rast("./fabdem_1.tif")

fabdem_1 <- rast("./Fabdem_v1/S24W071_FABDEM_V1-2.tif") 
fabdem_2 <- rast("./Fabdem_v1/S25W070_FABDEM_V1-2.tif")
fabdem_3 <- rast("./Fabdem_v1/S25W071_FABDEM_V1-2.tif")
fabdem_4 <- rast("./Fabdem_v1/S26W069_FABDEM_V1-2.tif")
fabdem_5 <- rast("./Fabdem_v1/S26W070_FABDEM_V1-2.tif")
fabdem_6 <- rast("./Fabdem_v1/S26W071_FABDEM_V1-2.tif")
fabdem_7 <- rast("./Fabdem_v1/S27W071_FABDEM_V1-2.tif")

## Proyectar el raster a UTM
taltal_utm <- project(taltal, crs_utm)

## Cargar el polígono
taltal_poligono <- vect("./La_negra_poligono.shp")

## Proyectar el polígono a UTM
taltal_poligono_utm <- project(taltal_poligono, crs_utm)

## Cargar los puntos

puntos_remociones <- vect("./puntos_remociones_poligonos.shp")
puntos_no_remociones <- vect("./puntos_no_remociones_poligonos.shp")

## Crear atributo que indica que son o no son remociones

puntos_remociones$REMOCION <- 1
puntos_no_remociones$REMOCION <- 0

## Limpiar de atributos innecesarios los archivos de puntos

puntos_remociones$rand_point <- NULL
puntos_remociones$fid <- NULL

puntos_no_remociones$rand_point <- NULL
puntos_no_remociones$rand_poi_1 <- NULL
puntos_no_remociones$fid <- NULL
puntos_no_remociones$Name <- NULL
puntos_no_remociones$descriptio <- NULL
puntos_no_remociones$timestamp <- NULL
puntos_no_remociones$begin <- NULL
puntos_no_remociones$end <- NULL
puntos_no_remociones$altitudeMo <- NULL
puntos_no_remociones$tessellate <- NULL
puntos_no_remociones$extrude <- NULL
puntos_no_remociones$visibility <- NULL
puntos_no_remociones$drawOrder <- NULL
puntos_no_remociones$icon <- NULL

## Unir los dos archivos de puntos

puntos <- vect(union(puntos_remociones,puntos_no_remociones))

# INICIALIZACIÓN DE SAGA-GIS

## Inicializar el programa
saga <- saga_gis(raster_backend = "terra",vector_backend = "SpatVector")

# Lista de objetos fabdem
fabdem_list <- list(fabdem_1, fabdem_2, fabdem_3, fabdem_4, fabdem_5, fabdem_6, fabdem_7)

## Aplicar basic_terrain_analysis a cada elemento de la lista
terrain_list <- lapply(fabdem_list, function(x) saga$ta_compound$basic_terrain_analysis(x))

## Elevation

combined_elevation <- do.call(merge, fabdem_list)

elevation <- mask(crop(combined_elevation,taltal_poligono),taltal_poligono)

writeRaster(elevation, "./fabdem_satelitales/elevation_v1.tif")

## Aspect

aspect_list <- lapply(terrain_list, function(terrain) {
  aspect <- terrain$aspect
  return(aspect)
})

combined_aspect <- do.call(merge, aspect_list)

aspect <- mask(crop(combined_aspect,taltal_poligono),taltal_poligono)

writeRaster(aspect, "./fabdem_satelitales/aspect_v1.tif")

## Hillshade

hillshade_list <- lapply(terrain_list, function(terrain) {
  hillshade <- terrain$shade
  return(hillshade)
})

combined_hillshade <- do.call(merge, hillshade_list)

hillshade <- mask(crop(combined_hillshade,taltal_poligono),taltal_poligono)

writeRaster(hillshade, "./fabdem_satelitales/hillshade_v1.tif")

## Slope

slope_list <- lapply(terrain_list, function(terrain) {
  slope <- terrain$slope
  return(slope)
})

combined_slope <- do.call(merge, slope_list)

slope <- mask(crop(combined_slope,taltal_poligono),taltal_poligono)

writeRaster(slope, "./fabdem_satelitales/slope_v1.tif",overwrite=TRUE)

## Hcurv

hcurv_list <- lapply(terrain_list, function(terrain) {
  hcurv <- terrain$hcurv
  return(hcurv)
})

combined_hcurv <- do.call(merge, hcurv_list)

hcurv <- mask(crop(combined_hcurv,taltal_poligono),taltal_poligono)

writeRaster(hcurv, "./fabdem_satelitales/hcurv_v1.tif")

## Vcurv

vcurv_list <- lapply(terrain_list, function(terrain) {
  vcurv <- terrain$vcurv
  return(vcurv)
})

combined_vcurv <- do.call(merge, vcurv_list)

vcurv <- mask(crop(combined_vcurv,taltal_poligono),taltal_poligono)

writeRaster(vcurv, "./fabdem_satelitales/vcurv_v1.tif")

## Convergence

convergence_list <- lapply(terrain_list, function(terrain) {
  convergence <- terrain$convergence
  return(convergence)
})

combined_convergence <- do.call(merge, convergence_list)

convergence <- mask(crop(combined_convergence,taltal_poligono),taltal_poligono)

writeRaster(convergence, "./fabdem_satelitales/convergence_v1.tif")

## Wetness

wetness_list <- lapply(terrain_list, function(terrain) {
  wetness <- terrain$wetness
  return(wetness)
})

combined_wetness <- do.call(merge, wetness_list)

wetness <- mask(crop(combined_wetness,taltal_poligono),taltal_poligono)

writeRaster(wetness, "./fabdem_satelitales/wetness_v1.tif")

## Lsfactor

lsfactor_list <- lapply(terrain_list, function(terrain) {
  lsfactor <- terrain$lsfactor
  return(lsfactor)
})

combined_lsfactor <- do.call(merge, lsfactor_list)

lsfactor <- mask(crop(combined_lsfactor,taltal_poligono),taltal_poligono)

writeRaster(lsfactor, "./fabdem_satelitales/lsfactor_v1.tif")

## Vall_depth

vall_depth_list <- lapply(terrain_list, function(terrain) {
  vall_depth <- terrain$vall_depth
  return(vall_depth)
})

combined_vall_depth <- do.call(merge, vall_depth_list)

vall_depth <- mask(crop(combined_vall_depth,taltal_poligono),taltal_poligono)

writeRaster(vall_depth, "./fabdem_satelitales/vall_depth_v1.tif")

## Otros que están en SAGA

### Melton

melton_full_list <- lapply(fabdem_list, function(x) saga$ta_hydrology$melton_ruggedness_number(x))

melton_list <- lapply(melton_full_list, function(melton) {
  melton <- melton$mrn
  return(melton)
})

combined_melton <- do.call(merge, melton_list)

melton <- mask(crop(combined_melton,taltal_poligono),taltal_poligono)

writeRaster(melton, "./fabdem_satelitales/melton_v1.tif")

### TRI

tri_list <- lapply(fabdem_list, function(x) saga$ta_morphometry$terrain_ruggedness_index_tri(x))

combined_tri <- do.call(merge, tri_list)

tri <- mask(crop(combined_tri,taltal_poligono),taltal_poligono)

writeRaster(tri,"./fabdem_satelitales/tri_v1.tif")

### TPI

tpi_list <- lapply(fabdem_list, function(x) saga$ta_morphometry$multi_scale_topographic_position_index_tpi(x))

combined_tpi <- do.call(merge, tpi_list)

tpi <- mask(crop(combined_tpi,taltal_poligono),taltal_poligono)

writeRaster(tpi,"./fabdem_satelitales/tpi_v1.tif",overwrite=TRUE)

### Solar

solar_list <- lapply(fabdem_list, function(x) saga$ta_lighting$potential_incoming_solar_radiation(x))
  

# Derivados de Whitebox

### Plan Curvature

wd <- "."

wbt_plan_curvature(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/plan_curvature_v1.tif"))

plan_curvature <- rast("./fabdem_satelitales/plan_curvature_v1.tif")

wbt_downslope_index(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/downslope_index_v1.tif"))

downslope_index <- rast("./fabdem_satelitales/downslope_index_v1.tif")

wbt_edge_density(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/edge_density_v1.tif"))

edge_density <- rast("./fabdem_satelitales/edge_density_v1.tif")

wbt_gaussian_curvature(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/gaussian_curvature_v1.tif"))

gaussian_curvature <- rast("./fabdem_satelitales/gaussian_curvature_v1.tif")

wbt_geomorphons(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/geomorphons_v1.tif"))

geomorphons <- rast("./fabdem_satelitales/geomorphons_v1.tif")

wbt_hypsometrically_tinted_hillshade(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/hypsometrically_tinted_hillshade_v1.tif"))

hypsometrically_tinted_hillshade <- rast("./fabdem_satelitales/hypsometrically_tinted_hillshade_v1.tif")

wbt_maximal_curvature(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/maximal_curvature_v1.tif"))

maximal_curvature <- rast("./fabdem_satelitales/maximal_curvature_v1.tif")

wbt_plan_curvature(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/plan_curvature_v1.tif"))

plan_curvature <- rast("./fabdem_satelitales/plan_curvature_v1.tif")

wbt_relative_topographic_position(dem=file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/relative_topographic_position_v1.tif"))

relative_topographic_position <- rast("./fabdem_satelitales/relative_topographic_position_v1.tif")

### Cálculo del área de atrape

wbt_fill_depressions(dem = file.path(wd, "./fabdem_satelitales/elevation_v1.tif"), output = file.path(wd, "./fabdem_satelitales/elevation_fill_v1.tif"))

wbt_d_inf_flow_accumulation(input = file.path(wd, "./fabdem_satelitales/elevation_fill_v1.tif"), output = file.path(wd, "./fabdem_satelitales/sca_v1.tif"))

sca <- rast("./fabdem_satelitales/sca_v1.tif")

### Cálculo del SPI

wbt_slope(dem=file.path(wd, "./fabdem_satelitales/elevation_fill_v1.tif"), output = file.path(wd, "./fabdem_satelitales/slope_v2.tif"))

pendiente <- rast("./fabdem_satelitales/slope_v2.tif")

wbt_stream_power_index(sca = file.path(wd, "./fabdem_satelitales/sca_v1.tif"), slope = file.path(wd, "./fabdem_satelitales/slope_v2.tif"), output = file.path(wd, "./fabdem_satelitales/spi_v1.tif"))

SPI <- rast("./fabdem_satelitales/spi_v1.tif")

### Satelitales

B1_001077 <- rast("./fabdem_satelitales/LC09_L2SP_001077_20231117_20231118_02_T1_SR_B1.TIF")
B2_001077 <- rast("./fabdem_satelitales/LC09_L2SP_001077_20231117_20231118_02_T1_SR_B2.TIF")
B3_001077 <- rast("./fabdem_satelitales/LC09_L2SP_001077_20231117_20231118_02_T1_SR_B3.TIF")
B4_001077 <- rast("./fabdem_satelitales/LC09_L2SP_001077_20231117_20231118_02_T1_SR_B4.TIF")
B5_001077 <- rast("./fabdem_satelitales/LC09_L2SP_001077_20231117_20231118_02_T1_SR_B5.TIF")
B6_001077 <- rast("./fabdem_satelitales/LC09_L2SP_001077_20231117_20231118_02_T1_SR_B6.TIF")
B7_001077 <- rast("./fabdem_satelitales/LC09_L2SP_001077_20231117_20231118_02_T1_SR_B7.TIF")

B1_001078 <- rast("./fabdem_satelitales/LC09_L2SP_001078_20231117_20231118_02_T1_SR_B1.TIF")
B2_001078 <- rast("./fabdem_satelitales/LC09_L2SP_001078_20231117_20231118_02_T1_SR_B2.TIF")
B3_001078 <- rast("./fabdem_satelitales/LC09_L2SP_001078_20231117_20231118_02_T1_SR_B3.TIF")
B4_001078 <- rast("./fabdem_satelitales/LC09_L2SP_001078_20231117_20231118_02_T1_SR_B4.TIF")
B5_001078 <- rast("./fabdem_satelitales/LC09_L2SP_001078_20231117_20231118_02_T1_SR_B5.TIF")
B6_001078 <- rast("./fabdem_satelitales/LC09_L2SP_001078_20231117_20231118_02_T1_SR_B6.TIF")
B7_001078 <- rast("./fabdem_satelitales/LC09_L2SP_001078_20231117_20231118_02_T1_SR_B7.TIF")

B1_233077 <- rast("./fabdem_satelitales/LC09_L2SP_233077_20231126_20231128_02_T1_SR_B1.TIF")
B2_233077 <- rast("./fabdem_satelitales/LC09_L2SP_233077_20231126_20231128_02_T1_SR_B1.TIF")
B3_233077 <- rast("./fabdem_satelitales/LC09_L2SP_233077_20231126_20231128_02_T1_SR_B1.TIF")
B4_233077 <- rast("./fabdem_satelitales/LC09_L2SP_233077_20231126_20231128_02_T1_SR_B1.TIF")
B5_233077 <- rast("./fabdem_satelitales/LC09_L2SP_233077_20231126_20231128_02_T1_SR_B1.TIF")
B6_233077 <- rast("./fabdem_satelitales/LC09_L2SP_233077_20231126_20231128_02_T1_SR_B1.TIF")
B7_233077 <- rast("./fabdem_satelitales/LC09_L2SP_233077_20231126_20231128_02_T1_SR_B1.TIF")

B1_233078 <- rast("./fabdem_satelitales/LC09_L2SP_233078_20231126_20231128_02_T1_SR_B1.TIF")
B2_233078 <- rast("./fabdem_satelitales/LC09_L2SP_233078_20231126_20231128_02_T1_SR_B1.TIF")
B3_233078 <- rast("./fabdem_satelitales/LC09_L2SP_233078_20231126_20231128_02_T1_SR_B1.TIF")
B4_233078 <- rast("./fabdem_satelitales/LC09_L2SP_233078_20231126_20231128_02_T1_SR_B1.TIF")
B5_233078 <- rast("./fabdem_satelitales/LC09_L2SP_233078_20231126_20231128_02_T1_SR_B1.TIF")
B6_233078 <- rast("./fabdem_satelitales/LC09_L2SP_233078_20231126_20231128_02_T1_SR_B1.TIF")
B7_233078 <- rast("./fabdem_satelitales/LC09_L2SP_233078_20231126_20231128_02_T1_SR_B1.TIF")

B1_list <- list(B1_001077, B1_001078, B1_233077, B1_233078)
combined_B1 <- do.call(merge, B1_list)
B1 <- mask(crop(combined_B1,taltal_poligono_utm),taltal_poligono_utm)

B2_list <- list(B2_001077, B2_001078, B2_233077, B2_233078)
combined_B2 <- do.call(merge, B2_list)
B2 <- mask(crop(combined_B2,taltal_poligono_utm),taltal_poligono_utm)

B3_list <- list(B3_001077, B3_001078, B3_233077, B3_233078)
combined_B3 <- do.call(merge, B3_list)
B3 <- mask(crop(combined_B3,taltal_poligono_utm),taltal_poligono_utm)

B4_list <- list(B4_001077, B4_001078, B4_233077, B4_233078)
combined_B4 <- do.call(merge, B4_list)
B4 <- mask(crop(combined_B4,taltal_poligono_utm),taltal_poligono_utm)

B5_list <- list(B5_001077, B5_001078, B5_233077, B5_233078)
combined_B5 <- do.call(merge, B5_list)
B5 <- mask(crop(combined_B5,taltal_poligono_utm),taltal_poligono_utm)

B6_list <- list(B6_001077, B6_001078, B6_233077, B6_233078)
combined_B6 <- do.call(merge, B6_list)
B6 <- mask(crop(combined_B6,taltal_poligono_utm),taltal_poligono_utm)

B7_list <- list(B7_001077, B7_001078, B7_233077, B7_233078)
combined_B7 <- do.call(merge, B7_list)
B7 <- mask(crop(combined_B7,taltal_poligono_utm),taltal_poligono_utm)

crs_longlat <- "+init=epsg:4326"

puntos_longlat <- project(puntos,crs_longlat)

writeVector(puntos_longlat,"./fabdem_satelitales/puntos.gpkg")

B1_longlat <- project(B1,crs_longlat)
B2_longlat <- project(B2,crs_longlat)
B3_longlat <- project(B3,crs_longlat)
B4_longlat <- project(B4,crs_longlat)
B5_longlat <- project(B5,crs_longlat)
B6_longlat <- project(B6,crs_longlat)
B7_longlat <- project(B7,crs_longlat)

writeRaster(B1_longlat,"./fabdem_satelitales/B1.tif",overwrite=TRUE)
writeRaster(B2_longlat,"./fabdem_satelitales/B2.tif",overwrite=TRUE)
writeRaster(B3_longlat,"./fabdem_satelitales/B3.tif",overwrite=TRUE)
writeRaster(B4_longlat,"./fabdem_satelitales/B4.tif",overwrite=TRUE)
writeRaster(B5_longlat,"./fabdem_satelitales/B5.tif",overwrite=TRUE)
writeRaster(B6_longlat,"./fabdem_satelitales/B6.tif",overwrite=TRUE)
writeRaster(B7_longlat,"./fabdem_satelitales/B7.tif",overwrite=TRUE)


# Cargar bandas

B2_np <- rast("./B2.tiff")
B3_np <- rast("./B3.tiff")
B4_np <- rast("./B4.tiff")
B5_np <- rast("./B5.tiff")
B6_np <- rast("./B6.tiff")
B7_np <- rast("./B6.tiff")

# Reproyectar bandas

B2_full <- project(B2_np, crs_utm)
B3_full <- project(B3_np, crs_utm)
B4_full <- project(B4_np, crs_utm)
B5_full <- project(B5_np, crs_utm)
B6_full <- project(B6_np, crs_utm)
B7_full <- project(B7_np, crs_utm)

# Enmascarar y cortar

B2 <- mask(crop(B2_full,poligono_utm),poligono_utm)
B3 <- mask(crop(B3_full,poligono_utm),poligono_utm)
B4 <- mask(crop(B4_full,poligono_utm),poligono_utm)
B5 <- mask(crop(B5_full,poligono_utm),poligono_utm)
B6 <- mask(crop(B6_full,poligono_utm),poligono_utm)
B7 <- mask(crop(B7_full,poligono_utm),poligono_utm)

# Cálculo de cada banda

ndvi <- (B5-B4)/(B5+B4)
gndvi <-(B5-B3)/(B5+B3)
evi <- 2.5*((B5-B4)/(B5+6*B4-7.5*B2+1))
ndmi <- (B5-B6)/(B5+B6)
bsi <- ((B6+B4)-(B5+B2))/((B6+B4)+(B5+B2))
ndwi <- (B3-B5)/(B3+B5)
ndgi <- (B3-B4)/(B3+B4)
nbri <- (B5-B7)/(B5+B7)

# Carga de características

puntos <- vect("./fabdem_satelitales/puntos.gpkg")
taltal_poligono <- vect("./La_negra_poligono.shp")

elevation <- rast("./fabdem_satelitales/elevation.tif")
aspect <- rast("./fabdem_satelitales/aspect.tif")
convergence <- rast("./fabdem_satelitales/convergence.tif")
downslope_index <- rast("./fabdem_satelitales/downslope_index.tif")
edge_density <- rast("./fabdem_satelitales/edge_density.tif")
gaussian_curvature <- rast("./fabdem_satelitales/gaussian_curvature.tif")
geomorphons <- rast("./fabdem_satelitales/geomorphons.tif")
hcurv <- rast("./fabdem_satelitales/hcurv.tif")
hillshade <- rast("./fabdem_satelitales/hillshade.tif")
lsfactor <- rast("./fabdem_satelitales/lsfactor.tif")
maximal_curvature <- rast("./fabdem_satelitales/maximal_curvature.tif")
melton <- rast("./fabdem_satelitales/melton.tif")
plan_curvature <- rast("./fabdem_satelitales/plan_curvature.tif")
relative_topographic_position <- rast("./fabdem_satelitales/relative_topographic_position.tif")
slope <- rast("./fabdem_satelitales/slope_v1.tif")
spi <- rast("./fabdem_satelitales/spi.tif")
tpi <- rast("./fabdem_satelitales/tpi.tif")
tri <- rast("./fabdem_satelitales/tri.tif")
vall_depth <- rast("./fabdem_satelitales/vall_depth.tif")
vcurv <- rast("./fabdem_satelitales/vcurv.tif")
wetness <- rast("./fabdem_satelitales/wetness.tif")

B1 <- rast("./fabdem_satelitales/B1.tif")
B2 <- rast("./fabdem_satelitales/B2.tif")
B3 <- rast("./fabdem_satelitales/B3.tif")
B4 <- rast("./fabdem_satelitales/B4.tif")
B5 <- rast("./fabdem_satelitales/B5.tif")
B6 <- rast("./fabdem_satelitales/B6.tif")
B7 <- rast("./fabdem_satelitales/B7.tif")

# Creación de stack y dataframe

pila <- c(aspect,elevation,hillshade,convergence,hcurv,lsfactor,
          melton,slope,plan_curvature,tpi,tri,vall_depth,vcurv,wetness)

pila1 <- c(B1,B2,B3,B4,B5,B6,B7)

pila2 <- c(downslope_index,edge_density,gaussian_curvature,geomorphons,maximal_curvature,relative_topographic_position,
           SPI)

rasValue <- terra::extract(pila, puntos)

rasValue1 <- terra::extract(pila1, puntos)

rasValue2 <- terra::extract(pila2, puntos)

valores <- cbind(as.data.frame(geom(puntos)),puntos$REMOCION,rasValue,rasValue1,rasValue2)

df <- as.data.table(valores)

# Limpieza del dataframe

df$geom <- NULL
df$part <- NULL
df$hole <- NULL
df$ID <- NULL
df$ID <- NULL
df$ID <- NULL

# Cambiar los nombres

names(df)[3] <- "REMOCION"
names(df)[4] <- "aspect"
names(df)[5] <- "elevation"
names(df)[6] <- "hillshade"
names(df)[7] <- "convergence"
names(df)[8] <- "hcurv"
names(df)[9] <- "lsfactor"
names(df)[10] <- "melton"
names(df)[11] <- "slope"
names(df)[12] <- "plan_curvature"
names(df)[13] <- "tpi"
names(df)[14] <- "tri"
names(df)[15] <- "vall_depth"
names(df)[16] <- "vcurv"
names(df)[17] <- "wetness"
names(df)[18] <- "B1"
names(df)[19] <- "B2"
names(df)[20] <- "B3"
names(df)[21] <- "B4"
names(df)[22] <- "B5"
names(df)[23] <- "B6"
names(df)[24] <- "B7"
names(df)[25] <- "downslope_index"
names(df)[26] <- "edge_density"
names(df)[27] <- "gaussian_curvature"
names(df)[28] <- "geomorphons"
names(df)[29] <- "maximal_curvature"
names(df)[30] <- "relative_topographic_position"
names(df)[31] <- "spi"

# Eliminar los NA

df <- na.omit(df)

# Hacer que sean todos DOUBLE

df$geomorphons <- as.double(df$geomorphons)

# Guardar el conjunto de datos

write.csv(df,"./df_da_v1_taltal.csv")

df <- read.csv("./df_da.csv")

# Crear columnas de índices satelitales

df$afri1600 <- (df$B5-0.66*df$B6)/(df$B5+0.66*df$B6)
df$afri2100 <- (df$B5-0.5*df$B7)/(df$B5+0.5*df$B7)
df$andwi <- (df$B2+df$B3+df$B4-df$B5-df$B6-df$B7)/(df$B2+df$B3+df$B4+df$B5+df$B6+df$B7)
df$arvi <- (df$B5 - (df$B4 - 1.0 * (df$B4 - df$B2))) / (df$B5 + (df$B4 - 1.0 * (df$B4 - df$B2)))
df$aweinsh <- 4.0 * (df$B2 - df$B6) - 0.25 * df$B5 + 2.75 * df$B7
df$aweish <- df$B2 + 2.5 * df$B3 - 1.5 * (df$B5 + df$B6) - 0.25 * df$B7
df$bai <- 1.0 / ((0.1 - df$B4)^2.0 + (0.06 - df$B5)^2.0)
df$bcc <- df$B2 / (df$B4 + df$B3 + df$B2)
df$bi <- ((df$B6 + df$B4) - (df$B5 + df$B2))/((df$B6 + df$B4) + (df$B5 + df$B2))
df$bitm <- (((df$B2^2.0)+(df$B3^2.0)+(df$B4^2.0))/3.0)^0.5
df$bixs <- (((df$B3^2.0)+(df$B4^2.0))/2.0)^0.5
df$blfei <- (((df$B3+df$B4+df$B7)/3.0)-df$B6)/(((df$B3+df$B4+df$B7)/3.0)+df$B6)
df$bndvi <- (df$B5 - df$B2)/(df$B5 + df$B2)
df$brba <- 	df$B4/df$B6
df$bwdrvi <- 	(1.0 * df$B5 - df$B2) / (1.0 * df$B5 + df$B2)
df$bal <- df$B4 + df$B6 - df$B5
df$cig <- (df$B5 / df$B3) - 1.0
df$csi <- df$B5/df$B7
df$cvi <- (df$B5 * df$B4) / (df$B3^2.0)
df$dbsi <- ((df$B6 - df$B3)/(df$B6 + df$B3)) - ((df$B5 - df$B4)/(df$B5 + df$B4))
df$dsi <- df$B6/df$B5
df$dswi1 <- df$B5/df$B6
df$dswi2 <- df$B6/df$B3
df$dswi3 <- df$B6/df$B4
df$dswi4 <- df$B3/df$B4
df$dswi5 <- (df$B5 + df$B3)/(df$B6 + df$B4)
df$dvi <- df$B5 - df$B4
df$ebi <- (df$B4 + df$B3 + df$B2)/((df$B3/df$B2) * (df$B4 - df$B2 + 1.0))
df$embi <- ((((df$B6 - df$B7 - df$B5)/(df$B6 + df$B7 + df$B5)) + 0.5) - ((df$B3 - df$B6)/(df$B3 + df$B6)) - 0.5)/((((df$B6 - df$B7 - df$B5)/(df$B6 + df$B7 + df$B5)) + 0.5) + ((df$B3 - df$B6)/(df$B3 + df$B6)) + 1.5)
df$evi <- 2.5*((df$B5-df$B4)/(df$B5+6*df$B4-7.5*df$B2+1))
df$exg <- 2 * df$B3 - df$B4 - df$B2
df$exgr <- (2.0 * df$B3 - df$B4 - df$B2) - (1.3 * df$B4 - df$B3)
df$fcvi <- df$B5 - ((df$B4 + df$B3 + df$B2)/3.0)
df$gari <- (df$B5 - (df$B3 - (df$B2 - df$B4))) / (df$B5 - (df$B3 + (df$B2 - df$B4)))
df$gbndvi <- (df$B5 - (df$B3 + df$B2))/(df$B5 + (df$B3 + df$B2))
df$gcc <- df$B3 / (df$B4 + df$B3 + df$B2)
df$gemi <- ((2.0*((df$B5^2.0)-(df$B4^2.0)) + 1.5*df$B5 + 0.5*df$B4)/(df$B5 + df$B4 + 0.5))*(1.0 - 0.25*((2.0 * ((df$B5^2.0) - (df$B4^2)) + 1.5 * df$B5 + 0.5 * df$B4)/(df$B5 + df$B4 + 0.5)))-((df$B4 - 0.125)/(1 - df$B4))
df$gli <- (2.0 * df$B3 - df$B4 - df$B2) / (2.0 * df$B3 + df$B4 + df$B2)
df$gndvi <- (df$B5 - df$B3)/(df$B5 + df$B3)
df$gosavi <- (df$B5 - df$B3)/(df$B5 + df$B3 + 0.16)
df$grndvi <- (df$B5 - (df$B3 + df$B4))/(df$B5 + (df$B3 + df$B4))
df$grvi <- df$B5/df$B3
df$gvmi <- ((df$B5 + 0.1) - (df$B7 + 0.02)) / ((df$B5 + 0.1) + (df$B7 + 0.02))
df$ikaw <- (df$B4 - df$B2)/(df$B4 + df$B2)
df$ipvi <- df$B5/(df$B5 + df$B4)
df$lswi <- (df$B5 - df$B6)/(df$B5 + df$B6)
df$mbi <- ((df$B6 - df$B7 - df$B5)/(df$B6 + df$B7 + df$B5)) + 0.5
df$mcari1 <- 	1.2 * (2.5 * (df$B5 - df$B4) - 1.3 * (df$B5 - df$B3))
df$mcari2 <- (1.5 * (2.5 * (df$B5 - df$B4) - 1.3 * (df$B5 - df$B3))) / ((((2.0 * df$B5 + 1) ^ 2) - (6.0 * df$B5 - 5 * (df$B4 ^ 0.5)) - 0.5) ^ 0.5)

df$mgrvi <- (df$B3 ^ 2.0 - df$B4 ^ 2.0) / (df$B3 ^ 2.0 + df$B4 ^ 2.0)
df$mirbi <-  10.0 * df$B7 - 9.8 * df$B6 + 2.0
df$mlswi26 <- (1.0 - df$B5 - df$B6)/(1.0 - df$B5 + df$B6)
df$mlswi27 <- (1.0 - df$B5 - df$B7)/(1.0 - df$B5 + df$B7)
df$mndvi <- (df$B5 - df$B7)/(df$B5 + df$B7)
df$mndwi <- (df$B3 - df$B6) / (df$B3 + df$B6)
df$mrbvi <- (df$B4 ^ 2.0 - df$B2 ^ 2.0)/(df$B4 ^ 2.0 + df$B2 ^ 2.0)
df$msavi <- 0.5 * (2.0 * df$B5 + 1 - (((2 * df$B5 + 1) ^ 2) - 8 * (df$B5 - df$B4)) ^ 0.5)
df$msi <- df$B6/df$B5
df$msr <- (df$B5 / df$B4 - 1) / ((df$B5 / df$B4 + 1) ^ 0.5)
df$mtvi1 <- 1.2 * (1.2 * (df$B5 - df$B3) - 2.5 * (df$B4 - df$B3))
df$mtvi2 <- 1.2 * (1.2 * (df$B5 - df$B3) - 2.5 * (df$B4 - df$B3))
df$muwir <- -4.0 * ((df$B2 - df$B3)/(df$B2 + df$B3)) + 2.0 * ((df$B3 - df$B5)/(df$B3 + df$B5)) + 2.0 * ((df$B3 - df$B7)/(df$B3 + df$B7)) - ((df$B3 - df$B6)/(df$B3 + df$B6))
df$nbai <- ((df$B7 - df$B6)/df$B3)/((df$B7 + df$B6)/df$B3)
df$nbr <- (df$B5 - df$B7) / (df$B5 + df$B7)
df$nbr2 <- (df$B6 - df$B7) / (df$B6 + df$B7)
df$nbrswir <- (df$B7 - df$B6 - 0.02)/(df$B7 + df$B6 + 0.1)
df$nbsims <- 0.36 * (df$B3 + df$B4 + df$B5) - (((df$B2 + df$B7)/df$B3) + df$B6)
df$ndbi <- (df$B6 - df$B5) / (df$B6 + df$B5)
df$nddi <- (((df$B5 - df$B4)/(df$B5 + df$B4)) - ((df$B3 - df$B5)/(df$B3 + df$B5)))/(((df$B5 - df$B4)/(df$B5 + df$B4)) + ((df$B3 - df$B5)/(df$B3 + df$B5)))
df$ndgi <- (df$B3 - df$B4)/(df$B3 + df$B4)
df$ndii <- (df$B5 - df$B6)/(df$B5 + df$B6)
df$ndponi <- (df$B6-df$B3)/(df$B6+df$B3)
df$ndsi <- 	(df$B3 - df$B6) / (df$B3 + df$B6)
df$ndsii <- (df$B3 - df$B5)/(df$B3 + df$B5)
df$ndswir <- (df$B5 - df$B6)/(df$B5 + df$B6)
df$ndsaii <- (df$B4 - df$B6) / (df$B4 + df$B6)
df$ndsoil <- 	(df$B7 - df$B3)/(df$B7 + df$B3)
df$ndti <- (df$B4-df$B3)/(df$B4+df$B3)
df$ndvi <- (df$B5 - df$B4)/(df$B5 + df$B4)
df$ndvimndwi <- ((df$B5 - df$B4)/(df$B5 + df$B4)) - ((df$B3 - df$B6)/(df$B3 + df$B6))
df$ndwi <- (df$B3 - df$B5) / (df$B3 + df$B5)
df$ndyi <- (df$B3 - df$B2) / (df$B3 + df$B2)
df$nirv <- ((df$B5 - df$B4) / (df$B5 + df$B4)) * df$B5
df$nli <- 	((df$B5 ^ 2) - df$B4)/((df$B5 ^ 2) + df$B4)
df$nmdi <- 	(df$B5 - (df$B6 - df$B7))/(df$B5 + (df$B6 - df$B7))
df$nrfig <- (df$B3 - df$B7) / (df$B3 + df$B7)
df$nrfir <- (df$B4 - df$B7) / (df$B4 + df$B7)
df$nsds <- (df$B6 - df$B7)/(df$B6 + df$B7)
df$nsdsi1 <- (df$B6-df$B7)/df$B6
df$nsdsi2 <- (df$B6-df$B7)/df$B7
df$nsdsi3 <- (df$B6-df$B7)/(df$B6+df$B7)
df$nwi <- (df$B2 - (df$B5 + df$B6 + df$B7))/(df$B2 + (df$B5 + df$B6 + df$B7))
df$normg <- 	df$B3/(df$B5 + df$B3 + df$B4)
df$normnir <- df$B5/(df$B5 + df$B3 + df$B4)
df$normr <- df$B4/(df$B5 + df$B3 + df$B4)
df$osavi <- (df$B5 - df$B4) / (df$B5 + df$B4 + 0.16)
df$pisi <- 	0.8192 * df$B2 - 0.5735 * df$B5 + 0.0750
df$rcc <- df$B4 / (df$B4 + df$B3 + df$B2)
df$rdvi <- 	(df$B5 - df$B4) / ((df$B5 + df$B4) ^ 0.5)
df$rgbvi <- (df$B3 ^ 2.0 - df$B2 * df$B4)/(df$B3 ^ 2.0 + df$B2 * df$B4)
df$rgri <- df$B4/df$B3
df$ri <- 	(df$B4 - df$B3)/(df$B4 + df$B3)
df$ri4xs <- (df$B4^2.0)/(df$B3^4.0)
df$s3 <- (df$B5 * (df$B4 - df$B6)) / ((df$B5 + df$B4) * (df$B5 + df$B6))
df$sarvi <- (1 + 1)*(df$B5 - (df$B4 - (df$B4 - df$B2))) / (df$B5 + (df$B4 - (df$B4 - df$B2)) + 1)
df$savi <- (1.0 + 1) * (df$B5 - df$B4) / (df$B5 + df$B4 + 1)
df$sipi <- 	(df$B5 - df$B1) / (df$B5 - df$B4)
df$slavi <- df$B5/(df$B4 + df$B7)
df$sr <- 	df$B5/df$B4
df$sr2 <- df$B5/df$B3
df$swi <- (df$B3 * (df$B5 - df$B6)) / ((df$B3 + df$B5) * (df$B5 + df$B6))
df$tdvi <- 	1.5 * ((df$B5 - df$B4)/((df$B5 ^ 2.0 + df$B4 + 0.5) ^ 0.5))
df$tgi <- 	- 0.5 * (190 * (df$B4 - df$B3) - 120 * (df$B4 - df$B2))
df$tvi <- 	(((df$B5 - df$B4)/(df$B5 + df$B4)) + 0.5) ^ 0.5
df$trivi <- 	0.5 * (120 * (df$B5 - df$B3) - 200 * (df$B4 - df$B3))
df$ui <- 	(df$B7 - df$B5)/(df$B7 + df$B5)
df$vari <- 	(df$B3 - df$B4) / (df$B3 + df$B4 - df$B2)
df$vibi <- ((df$B5-df$B4)/(df$B5+df$B4))/(((df$B5-df$B4)/(df$B5+df$B4)) + ((df$B6-df$B5)/(df$B6+df$B5)))
df$vgnirbi <- (df$B3 - df$B5)/(df$B3 + df$B5)
df$vrnirbi <- 	(df$B4 - df$B5)/(df$B4 + df$B5)
df$wi1 <- (df$B3 - df$B7) / (df$B3 + df$B7)
df$wi2 <- (df$B2 - df$B7) / (df$B2 + df$B7)
df$wi2015 <- 1.7204 + 171 * df$B3 + 3 * df$B4 - 70 * df$B5 - 45 * df$B6 - 71 * df$B7
df$wri <- 	(df$B3 + df$B4)/(df$B5 + df$B6)


write.csv(df,"./df_da_taltal_v2.csv")

df_da_taltal_v2 <- read.csv("./df_da_taltal_v2.csv")

df <- df_da_taltal_v2

# Modelo de Machine Learning

df$REMOCION <- as.factor(df$REMOCION)

df$ID <- NULL
df$X <- NULL

# Creación de la tarea ("Task")

df <- na.omit(df)

task = mlr3spatiotempcv::TaskClassifST$new(
  id = "remociones",
  backend = df,
  target = "REMOCION",
  coordinate_names = c("x", "y"),
  extra_args = list(
    coords_as_features = FALSE,
    crs = 4326)
)

# Importancia de factores

### AUC

set.seed(1)
filter = flt("auc")
auc_table <- as.data.table(filter$calculate(task))


### ANOVA

set.seed(2)
filter = flt("anova")
anova <- as.data.table(filter$calculate(task))


### CMIM

set.seed(3)
filter = flt("cmim")
cmim <- as.data.table(filter$calculate(task))

### DISR

set.seed(4)
filter = flt("disr")
disr <- as.data.table(filter$calculate(task))

### FIND CORRELATION

set.seed(5)
filter = flt("find_correlation")
corr <- as.data.table(filter$calculate(task))


### INFORMATION GAIN

set.seed(7)
filter = flt("information_gain")
ig <- as.data.table(filter$calculate(task))

### JMI

set.seed(8)
filter = flt("jmi")
jmi <- as.data.table(filter$calculate(task))

### JMIM

set.seed(9)
filter = flt("jmim")
jmim <- as.data.table(filter$calculate(task))

### MIM

set.seed(10)
filter = flt("mim")
mim <- as.data.table(filter$calculate(task))

### MRMR

set.seed(11)
filter = flt("mrmr")
mrmr <- as.data.table(filter$calculate(task))

### PERFORMANCE

set.seed(12)
filter = flt("performance")
performance <- as.data.table(filter$calculate(task))

### PERMUTATION (Se demora!)

set.seed(13)
filter = flt("permutation")
permutation <- as.data.table(filter$calculate(task))

### RELIEF

set.seed(14)
filter = flt("relief")
relief <- as.data.table(filter$calculate(task))

### VARIANCE

set.seed(16)
filter = flt("variance")
variance <- as.data.table(filter$calculate(task))

### IMPURITY

lrn = lrn("classif.ranger")
lrn$param_set$values = list(importance = "impurity")

filter = flt("importance", learner = lrn)
filter$calculate(task)
impurity <- as.data.table(filter)

## CFS ##

# Suponiendo que ya tienes las tablas de datos auc_table, anova, cmim, etc.
rankings_list <- list(auc_table, anova, cmim, disr, corr, ig, jmi, jmim, mim, mrmr, performance, permutation, relief, variance, impurity)

# Normalizar y calcular el ranking para cada tabla
normalize_and_rank <- function(data_table) {
  # Asumiendo que 'score' es la columna con los puntajes
  # Normalizar los puntajes
  data_table[, normalized_score := (score - min(score)) / (max(score) - min(score))]
  
  # Calcular el ranking basado en los puntajes normalizados
  data_table[, rank := frank(-normalized_score, ties.method = "average")]
  
  return(data_table[, .(feature, rank)])
}

# Aplicar la función a cada tabla de rankings
normalized_rankings_list <- lapply(rankings_list, normalize_and_rank)

# Función para calcular el ranking acumulativo
calculate_cumulative_ranking <- function(rankings_list) {
  # Unir todas las tablas en una sola
  combined_rankings <- rbindlist(rankings_list, fill = TRUE)
  
  # Reemplazar NA con el máximo ranking + 1
  max_rank <- max(combined_rankings$rank, na.rm = TRUE)
  combined_rankings[is.na(rank)]$rank <- max_rank + 1
  
  # Calcular el ranking acumulativo
  cumulative_ranking <- combined_rankings[, .(cumulative_rank = sum(rank)), by = .(feature)]
  
  # Ordenar los resultados
  cumulative_ranking <- cumulative_ranking[order(cumulative_rank)]
  
  return(cumulative_ranking)
}

# Calcular el ranking acumulativo
cumulative_ranking <- calculate_cumulative_ranking(normalized_rankings_list)

# Seleccionar los factores con el ranking acumulativo más bajo
selected_factors <- head(cumulative_ranking, n = 30)

print(selected_factors)

###############################

df1 <- data.frame(df$x,df$y,df$REMOCION,df$downslope_index,df$wetness,df$tri,df$edge_density,df$melton,df$exg,
                  df$slope,df$elevation,df$vari,df$hcurv,df$rgri,df$normg,df$gcc,df$tpi,
                  df$rgbvi,df$dswi4,df$vcurv,df$gli,df$mgrvi,df$tgi,df$vall_depth,df$muwir,df$exgr,df$gemi,
                  df$slavi,df$cvi,df$sr2,df$gndvi,df$sr,df$gbndvi)

names(df1) <- c("x","y","REMOCION","downslope_index","wetness","tri","edge_density","melton","exg","slope","elevation",
                "vari","hcurv","rgri","normg","gcc","tpi","rgbvi","dswi4","vcurv","gli","mgrvi","tgi","vall_depth","muwir",
                "exgr","gemi","slavi","cvi","sr2","gndvi","sr","gbnvi")

df2 <- df1

df2$REMOCION <- NULL
df2$x <- NULL
df2$y <- NULL

# Calcular la matriz de correlación
cor_matrix <- cor(df2)

# Crear y mostrar el gráfico de correlación
corrplot(cor_matrix, method = "number", type = "upper", tl.cex = 0.8)

df1$wetness <- NULL
df1$tri <- NULL
df1$gcc <- NULL
df1$rgbvi <- NULL
df1$gli <- NULL
df1$tgi <- NULL
df1$rgri <- NULL
df1$normg <- NULL
df1$dswi4 <- NULL
df1$mgrvi <- NULL
df1$tpi <- NULL
df1$cvi <- NULL
df1$sr2 <- NULL
df1$gndvi <- NULL
df1$sr <- NULL
df1$gbnvi <- NULL



##################################

# Dataset alternativo de prueba

df <- df_da_taltal_v1

df1 <- data.frame(df$x,df$y,df$REMOCION,df$edge_density,df$tri,df$slope,df$downslope_index,df$melton,df$wetness,df$exg,
                  df$hcurv,df$gaussian_curvature,df$gcc,df$grndvi,df$gemi,df$mirbi,df$nsdsi1,df$elevation,df$spi,
                  df$evi,df$mlswi26,df$gli,df$tpi)

names(df1) <- c("x","y","REMOCION","edge_density","tri","slope","downslope_index","melton","wetness","exg","hcurv",
                "gaussian_curvature","gcc","grndvi","gemi","mirbi","nsdsi1","elevation","spi","evi","mlswi26","gli","tpi")

df1$gli <- NULL
df1$nsdsi1 <- NULL
df1$wetness <- NULL
df1$downslope_index <- NULL
df1$tpi <- NULL
df1$gemi <- NULL
df1$gcc <- NULL
df1$slope <- NULL

df2 <- df1

df2$REMOCION <- NULL
df2$x <- NULL
df2$y <- NULL

df2$REMOCION <- as.factor(df2$REMOCION)

# Calcular la matriz de correlación
cor_matrix <- cor(df2)

# Crear y mostrar el gráfico de correlación
corrplot(cor_matrix, method = "number", type = "upper", tl.cex = 0.8)


##############################

df2$REMOCION <- as.factor(df2$REMOCION)

df1 <- df

task_v1 = mlr3spatiotempcv::TaskClassifST$new(
  id = "remociones",
  backend = df1,
  target = "REMOCION",
  coordinate_names = c("x", "y"),
  extra_args = list(
    coords_as_features = FALSE,
    crs = 4326)
)


## Súper benchmark con los 31 modelos

design = benchmark_grid(
  tasks = task_v1,
  learners = lrns(c("classif.AdaBoostM1", "classif.bart", "classif.C50", "classif.cforest", "classif.ctree",
                    "classif.debug", "classif.earth", "classif.featureless", "classif.fnn", "classif.gam",
                    "classif.gamboost",
                    "classif.gausspr","classif.glmboost", "classif.glmnet", "classif.IBk", "classif.JRip",
                    "classif.ksvm", "classif.liblinear","classif.lightgbm", "classif.LMT", "classif.log_reg",
                    "classif.multinom", "classif.naive_bayes", "classif.nnet", "classif.OneR", "classif.PART", "classif.randomForest",
                    "classif.ranger", "classif.rfsrc", "classif.rpart", "classif.svm", "classif.xgboost"),
                  predict_type = "prob", predict_sets = c("train", "test")
  ),
  resamplings = rsmps("cv", folds = 5)
)

bmr = benchmark(design)

## Agregar las mediciones

measures = list(
  msr("classif.auc", predict_sets = "train", id = "auc_train"),
  msr("classif.auc", id = "auc_test")
)

tab = bmr$aggregate(measures)

print(tab)

ranks = tab[, .(learner_id, rank_train = rank(-auc_test), rank_test = rank(-auc_test), auc_test= auc_test), by = task_id]
print(ranks)

# group by levels of learner_id, return columns:
# - mean rank of col 'rank_train' (per level of learner_id)
# - mean rank of col 'rank_test' (per level of learner_id)
ranks = ranks[, .(mrank_train = mean(rank_train), mrank_test = mean(rank_test), auc_test = auc_test), by = learner_id]

# print the final table, ordered by mean rank of AUC test
ranks[order(mrank_test)]

modelos <- ranks[order(mrank_test)]

##############################################

write.csv(df1,"./df1_hp_taltal.csv")

df1 <- read.csv("./df1_hp_taltal.csv")

df1$X <- NULL

df1$REMOCION <- as.factor(df1$REMOCION)


# AFINACIÓN DE HIPERPARÁMETROS

## "classif.AdaBoostM1"

### Explorar el espacio de parámetros

learner_adaboost = mlr3::lrn("classif.AdaBoostM1", predict_type = "prob")

learner_adaboost$param_set$ids()

learner_adaboost$param_set

search_space = ps(
  I = p_int(lower = 1, upper = 100),
  P = p_int(lower = 90, upper = 100),
  S = p_int(lower= 1, upper = 100)
)

# I, podrías probar valores como 50, 100, 200.
#Para P y Q, podrías considerar valores en un rango como 0.01, 0.1, 0.5, 1.
#Para S, podrías probar con diferentes proporciones del conjunto de datos, como 0.5, 0.7, 0.9.
#Para W, podrías experimentar con diferentes esquemas de ponderación.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_adaboost,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_adaboost$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

## "classif.bart"

learner_bart = mlr3::lrn("classif.bart", predict_type = "prob")

learner_bart$param_set$ids()

learner_bart$param_set

search_space = ps(
  ntree = p_int(lower = 1, upper = 300),
  k = p_dbl(lower = 0, upper = 10),
  power = p_dbl(lower= 0, upper = 3),
  base = p_dbl(lower= 0.9, upper = 1),
  numcut = p_int(lower = 1, upper = 100)
)

#Para ntree, podrías probar con valores como 50, 100, 200.
#Para k, considera rangos que van desde valores pequeños (como 1 o 2) hasta valores moderadamente altos (como 10).
#Para power, podrías explorar valores en un rango como 1, 2, 3.
#Para base, podrías considerar valores como 0.95, 0.99.
#Para numcut, valores típicos pueden ser 10, 20, 50, especialmente para variables continuas.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_bart,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_bart$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

## "classif.C50"

learner_c50 = mlr3::lrn("classif.C50", predict_type = "prob")

learner_c50$param_set$ids()

learner_c50$param_set

search_space = ps(
  trials = p_int(lower = 1, upper = 50),
  CF = p_dbl(lower = 0, upper = 1),
  minCases = p_int(lower= 1, upper = 20)
)

#trials (Número de Iteraciones de Boosting): Este parámetro controla cuántas veces se repite el proceso de boosting. Un número mayor puede mejorar el rendimiento, pero también aumenta el riesgo de sobreajuste. Podrías probar con valores como 1, 10, 20.
#rules: Determina si el modelo debe generar reglas en lugar de árboles. Puedes probar con valores booleanos (TRUE o FALSE).
#CF (Factor de Confianza para la Poda de Árboles): Este valor se utiliza para podar los árboles y evitar el sobreajuste. Valores típicos están en el rango de 0.1 a 0.25.
#minCases (Número Mínimo de Casos en un Nodo): Este parámetro controla el tamaño mínimo de los nodos de los árboles y puede afectar la profundidad de los árboles. Valores comunes pueden ser 1, 5, 10.
#winnow: Controla si se debe realizar una selección de características preliminar. También puedes probar con valores booleanos (TRUE o FALSE).
#noGlobalPruning: Indica si se debe desactivar la poda global del árbol. De nuevo, es un parámetro booleano.
#earlyStopping: Permite detener el entrenamiento si no hay mejora en el rendimiento del modelo, lo cual es útil para evitar el sobreajuste.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_c50,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_c50$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551


## "classif.ctree"

learner_ctree = mlr3::lrn("classif.ctree", predict_type = "prob")

learner_ctree$param_set$ids()

learner_ctree$param_set

learner_ctree$param_set$values$testtype = "MonteCarlo"

search_space = ps(
  minsplit = p_int(lower = 1, upper = 30),
  minbucket  = p_int(lower = 5, upper = 15),
  maxdepth = p_int(lower= 0, upper = 15),
  alpha = p_dbl(lower= 0.01, upper = 0.1),
  mincriterion = p_int(lower= 0, upper = 1),
  mtry = p_int(lower= 0, upper = 14),
  maxsurrogate  = p_int(lower= 0, upper = 5),
  nresample  = p_int(lower= 1000, upper = 10000)
)

#minsplit (Número Mínimo de Observaciones que se Deben Exceder para Dividir un Nodo): Controla la profundidad del árbol. Valores comunes pueden ser 10, 20, 30.
#minbucket (Tamaño Mínimo de los Nodos Finales): Similar a minsplit, pero para los nodos finales del árbol. Valores típicos podrían ser 5, 10, 15.
#maxdepth (Profundidad Máxima del Árbol): Controla la profundidad máxima de cada árbol. Puedes probar con valores como 5, 10, 15.
#alpha (Nivel de Significancia para el Test de División de Nodos): Este parámetro controla la significancia estadística necesaria para dividir un nodo. Valores comunes pueden estar en el rango de 0.01 a 0.1.
#mincriterion (Reducción Mínima en la Estadística de Prueba): Es un umbral para la mejora en el criterio de división. Puede variar, por ejemplo, de 0 a 1.
#mtry (Número de Variables Probadas en Cada División): Especialmente relevante si hay muchas variables. Podrías probar con valores como √(número de variables), 1/3 del número de variables.
#maxsurrogate (Número Máximo de Variables Sustitutas a Considerar): Útil en presencia de valores perdidos. Puedes probar con valores como 0, 2, 5.
#nresample (Número de Muestras para Permutaciones): Este parámetro es relevante para pruebas de permutación. Valores típicos pueden ser 1000, 5000.
#stump (Indica si se Debe Crear un Árbol de un Solo Nivel): Puede tomar valores booleanos (TRUE o FALSE).
#testtype (Tipo de Test para la División): Determina el tipo de prueba estadística utilizada para las divisiones. Explora diferentes opciones disponibles.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_ctree,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_ctree$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

## "classif.earth"

learner_earth = mlr3::lrn("classif.earth", predict_type = "prob")

learner_earth$param_set$ids()

learner_earth$param_set

search_space = ps(
  degree = p_int(lower = 1, upper = 2),
  penalty = p_int(lower = 2, upper = 4),
  thresh  = p_dbl(lower = 0.001, upper = 0.01),
  minspan = p_int(lower = 0, upper = 20),
  endspan = p_int(lower = 0, upper = 10),
  nprune = p_int(lower= 1, upper = 20)
)

#degree (Grado de las Interacciones): Determina el grado máximo de interacción entre las variables. Un valor común puede ser 1 (solo efectos principales) o 2 (interacciones de hasta dos vías).
#penalty (Penalización por Añadir un Término al Modelo): Afecta la complejidad del modelo. Valores comunes pueden estar en el rango de 2 a 4.
#nk (Número Máximo de Términos en el Modelo): Controla la cantidad máxima de términos (knots) que se pueden añadir al modelo. Valores típicos pueden ser 10, 20, 30.
#thresh (Umbral para la Mejora en el Modelo): Determina cuánta mejora se necesita para considerar un término adicional como significativo. Puede variar, por ejemplo, de 0.001 a 0.01.
#minspan (Número Mínimo de Datos en un Bin): Afecta la colocación de los nodos y puede controlar el sobreajuste. Valores comunes pueden ser 5, 10, 20.
#endspan (Número Mínimo de Datos en los Bins al Final de las Variables): Similar a minspan pero específico para los extremos de las variables. Valores típicos podrían ser 0, 5, 10.
#nprune (Número de Términos a Conservar Después de la Poda): Controla cuántos términos se conservan en el modelo final después del proceso de poda. Puede ser igual a nk o menor.
#fast.k y fast.beta (Parámetros para la Búsqueda Rápida): Estos parámetros controlan cómo se realiza la búsqueda rápida de términos. Podrías explorar diferentes combinaciones de estos valores.
#linpreds (Variables a Tratar como Lineales): Permite especificar qué variables deben entrar en el modelo solo como términos lineales.
#varmod.method (Método para Modelar la Varianza de los Errores): Útil si se sospecha heterocedasticidad en los errores.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_earth,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_earth$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

## "classif.gam"

learner_gam = mlr3::lrn("classif.gam", predict_type = "prob")

learner_gam$param_set$ids()

learner_gam$param_set

search_space = ps(
  gamma = p_dbl(lower = 1, upper = 10),
  epsilon = p_dbl(lower = 1e-07, upper = 1),
  maxit  = p_int(lower = 1, upper = 500),
  mgcv.tol = p_dbl(lower = 1e-07, upper = 1)

)

#method (Método de Suavizado): Controla el método de suavizado utilizado. Puedes elegir entre varios métodos como "REML", "GCV.Cp", "ML", etc.
#gamma (Parámetro de Ajuste para la Complejidad del Modelo): Un valor más alto penaliza modelos más complejos. Puedes probar valores como 1.0, 1.5, 2.0.
#scale (Escala de los Términos de Suavizado): Un valor de 0 implica que se estima de los datos. Puedes explorar otros valores para ver su efecto.
#select (Selección de Términos de Modelo): Un valor booleano que determina si se realiza la selección de términos de modelo.
#epsilon (Umbral de Convergencia para el Algoritmo de Ajuste): Controla la precisión del ajuste del modelo. Valores típicos pueden ser 1e-06, 1e-07.
#maxit (Número Máximo de Iteraciones para el Algoritmo de Ajuste): Define el número máximo de iteraciones. Puedes probar con valores como 100, 200, 300.
#mgcv.tol (Tolerancia para la Convergencia en mgcv): Similar a epsilon, pero específico para el paquete mgcv. Valores típicos pueden ser 1e-06, 1e-07.
#scale.est (Método para Estimar la Escala de los Residuos): Puedes elegir entre diferentes métodos como "fletcher", "gcv", "pml".
#nthreads (Número de Hilos para Computación Paralela): Útil si quieres aprovechar la computación paralela. Puedes probar con valores como 1, 2, 4, dependiendo de tu hardware.
#edge.correct (Corrección de Borde para Términos de Suavizado): Un valor booleano que determina si se aplica la corrección de borde en los términos de suavizado.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_gam,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_gam$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546


## "classif.gamboost"


learner_gamboost = mlr3::lrn("classif.gamboost", predict_type = "prob")

learner_gamboost$param_set$ids()

learner_gamboost$param_set

search_space = ps(
  dfbase = p_int(lower = 2, upper = 5),
  mstop = p_int(lower = 50, upper = 200),
  nu  = p_dbl(lower = 0.01, upper = 0.1)
)

#baselearner (Aprendiz Base): Determina el tipo de aprendiz base para el boosting. Las opciones típicas incluyen "bbs" (basis function boosting), entre otros. La elección depende del tipo de datos y el problema.
#dfbase (Grados de Libertad para el Aprendiz Base): Controla la complejidad del modelo base. Un número mayor permite una mayor flexibilidad. Valores comunes pueden ser 3, 4, 5.
#family (Familia de Distribuciones): Define la familia de distribución para el modelo, como "Binomial", "Gaussian", etc. La elección depende del tipo de respuesta que estás modelando (por ejemplo, binomial para respuestas binarias).
#link (Función de Enlace): La función de enlace para la familia de distribuciones seleccionada. Por ejemplo, "logit" es común para respuestas binomiales.
#mstop (Número de Pasos de Boosting): Define el número de iteraciones de boosting. Un número mayor puede mejorar la precisión, pero también aumenta el riesgo de sobreajuste. Valores típicos pueden ser 50, 100, 200.
#nu (Tasa de Aprendizaje): Es la tasa de aprendizaje del algoritmo de boosting. Un valor más pequeño conduce a un aprendizaje más lento y posiblemente a modelos más precisos. Valores comunes pueden ser 0.1, 0.01.
#risk (Función de Riesgo): Define la función de riesgo utilizada para el aprendizaje. Las opciones incluyen "inbag", "oobag" (out-of-bag), etc.
#type (Tipo de Boosting): Puede ser "adaboost" o "gentleboost". Cada tipo tiene sus propias características y puede funcionar mejor en diferentes conjuntos de datos.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_gamboost,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_gamboost$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797


## "classif.gausspr"

learner_gausspr = mlr3::lrn("classif.gausspr", predict_type = "prob")

learner_gausspr$param_set$ids()

learner_gausspr$param_set

search_space = ps(
  kernel = p_fct(levels = c("rbfdot", "polydot")),
  sigma = p_dbl(lower = -2, upper = 2, depends = (kernel == "rbfdot")),
  degree  = p_dbl(lower = 0.01, upper = 0.1, depends = (kernel == "polydot")),
  offset = p_dbl(lower = -2, upper = 2, depends = (kernel == "polydot")),
  tol = p_dbl(lower = 0.0001, upper = 1),
  fit = p_lgl()
)

#baselearner (Aprendiz Base): Determina el tipo de aprendiz base para el boosting. Las opciones típicas incluyen "bbs" (basis function boosting), entre otros. La elección depende del tipo de datos y el problema.
#dfbase (Grados de Libertad para el Aprendiz Base): Controla la complejidad del modelo base. Un número mayor permite una mayor flexibilidad. Valores comunes pueden ser 3, 4, 5.
#family (Familia de Distribuciones): Define la familia de distribución para el modelo, como "Binomial", "Gaussian", etc. La elección depende del tipo de respuesta que estás modelando (por ejemplo, binomial para respuestas binarias).
#link (Función de Enlace): La función de enlace para la familia de distribuciones seleccionada. Por ejemplo, "logit" es común para respuestas binomiales.
#mstop (Número de Pasos de Boosting): Define el número de iteraciones de boosting. Un número mayor puede mejorar la precisión, pero también aumenta el riesgo de sobreajuste. Valores típicos pueden ser 50, 100, 200.
#nu (Tasa de Aprendizaje): Es la tasa de aprendizaje del algoritmo de boosting. Un valor más pequeño conduce a un aprendizaje más lento y posiblemente a modelos más precisos. Valores comunes pueden ser 0.1, 0.01.
#risk (Función de Riesgo): Define la función de riesgo utilizada para el aprendizaje. Las opciones incluyen "inbag", "oobag" (out-of-bag), etc.
#type (Tipo de Boosting): Puede ser "adaboost" o "gentleboost". Cada tipo tiene sus propias características y puede funcionar mejor en diferentes conjuntos de datos.

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_gausspr,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_gausspr$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

## "classif.glmboost"

learner_glmboost = mlr3::lrn("classif.glmboost", predict_type = "prob")

learner_glmboost$param_set$ids()

learner_glmboost$param_set

search_space = ps(
  mstop = p_int(lower = 50, upper = 200),
  nu = p_dbl(lower = 0.01, upper = 0.1),
  risk = p_fct(levels = c("inbag", "oobag", "none"))
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_glmboost,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_glmboost$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

## "classif.glmnet"

learner_bart = mlr3::lrn("classif.bart", predict_type = "prob")
learner_c50 = mlr3::lrn("classif.C50", predict_type = "prob")
learner_ctree = mlr3::lrn("classif.ctree", predict_type = "prob")
learner_earth = mlr3::lrn("classif.earth", predict_type = "prob")
learner_gam = mlr3::lrn("classif.gam", predict_type = "prob")
learner_gamboost = mlr3::lrn("classif.gamboost", predict_type = "prob")
learner_gausspr = mlr3::lrn("classif.gausspr", predict_type = "prob")
learner_glmboost = mlr3::lrn("classif.glmboost", predict_type = "prob")
learner_glmnet = mlr3::lrn("classif.glmnet", predict_type = "prob")

learner_glmnet$param_set$ids()

learner_glmnet$param_set

search_space = ps(
  alpha = p_dbl(lower = 0, upper = 1),
  nlambda = p_int(lower = 20, upper = 100),
  standardize = p_lgl(),
  intercept = p_lgl()
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 1000)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_glmnet,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_glmnet$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962


## "classif.IBk"

learner_bart = mlr3::lrn("classif.bart", predict_type = "prob")
learner_c50 = mlr3::lrn("classif.C50", predict_type = "prob")
learner_ctree = mlr3::lrn("classif.ctree", predict_type = "prob")
learner_earth = mlr3::lrn("classif.earth", predict_type = "prob")
learner_gam = mlr3::lrn("classif.gam", predict_type = "prob")
learner_gamboost = mlr3::lrn("classif.gamboost", predict_type = "prob")
learner_gausspr = mlr3::lrn("classif.gausspr", predict_type = "prob")
learner_glmboost = mlr3::lrn("classif.glmboost", predict_type = "prob")
learner_glmnet = mlr3::lrn("classif.glmnet", predict_type = "prob")
learner_IBk = mlr3::lrn("classif.IBk", predict_type = "prob")

learner_IBk$param_set$ids()

learner_IBk$param_set$value$A

search_space = ps(
  K= p_int(lower = 1, upper = 30),
  W = p_int(lower = 0, upper = 20),
  I = p_lgl()
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_IBk,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_IBk$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

## "classif.ksvm"


learner_ksvm = mlr3::lrn("classif.ksvm", predict_type = "prob")

learner_ksvm$param_set$ids()

learner_ksvm$param_set

search_space = ps(
  kernel = p_fct(levels = c("rbfdot", "polydot")),
  sigma = p_dbl(lower = 0.01, upper = 1, depends = (kernel == "rbfdot")),
  degree  = p_int(lower = 1, upper = 5, depends = (kernel == "polydot"))
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_ksvm,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_ksvm$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162


## "classif.lightgbm"


learner_lightgbm = mlr3::lrn("classif.lightgbm", predict_type = "prob")

learner_lightgbm$param_set$ids()

learner_lightgbm$param_set

search_space = ps(
  learning_rate = p_dbl(lower = 0.2, upper = 0.74),
  num_leaves = p_int(lower = 120, upper = 180),
  feature_fraction = p_dbl(lower= 0.22, upper = 0.42),
  bagging_fraction = p_dbl(lower= 0.55, upper = 0.72),
  bagging_freq = p_int(lower= 20, upper = 100),
  max_depth = p_int(lower = 350, upper = 670),
  min_data_in_leaf = p_int(lower = 12, upper = 48)
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_lightgbm,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_lightgbm$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

## "classif.LMT"

learner_LMT = mlr3::lrn("classif.LMT", predict_type = "prob")

learner_LMT$param_set$ids()

learner_LMT$param_set$values$C <- FALSE

search_space = ps(
  M = p_int(lower = 1, upper = 20),
  W = p_dbl(lower = 0, upper = 0.5),
  I = p_int(lower= 5, upper = 10),
  P = p_lgl(),
  R = p_lgl()
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_LMT,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_LMT$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

## "classif.log_reg"

learner_log_reg = mlr3::lrn("classif.log_reg", predict_type = "prob")

learner_log_reg$param_set$ids()

learner_log_reg$param_set

search_space = ps(
  epsilon = p_dbl(lower = 1e-10, upper = 1e-6),
  maxit = p_int(lower = 10, upper = 100),
  trace = p_lgl()
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_log_reg,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_log_reg$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939

## "classif.naive_bayes"

learner_naive_bayes = mlr3::lrn("classif.naive_bayes", predict_type = "prob")

learner_naive_bayes$param_set$ids()

learner_naive_bayes$param_set

search_space = ps(
  eps = p_dbl(lower = 1e-10, upper = 1e-5),
  laplace= p_dbl(lower = 0, upper = 5),
  threshold = p_dbl(lower = 0.001, upper = 0.1)
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_naive_bayes,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_naive_bayes$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939


#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

## "classif.nnet"

learner_nnet = mlr3::lrn("classif.nnet", predict_type = "prob")

learner_nnet$param_set$ids()

learner_nnet$param_set

search_space = ps(
  decay = p_dbl(lower = 0, upper = 0.1),
  maxit= p_int(lower = 50, upper = 500),
  size= p_int(lower = 1, upper = 10),
  rang = p_dbl(lower = 0.1, upper = 1),
  abstol = p_dbl(lower = 1e-6, upper = 1e-4),
  reltol = p_dbl(lower = 1e-6, upper = 1e-4)
  
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_nnet,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_nnet$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939


#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

#     decay maxit size      rang       abstol       reltol learner_param_vals  x_domain classif.auc
# 0.0551435   255   10 0.5080007 6.807949e-05 5.769071e-05          <list[6]> <list[6]>   0.8695175

## "classif.PART"

learner_PART = mlr3::lrn("classif.PART", predict_type = "prob")

learner_PART$param_set$ids()

learner_PART$param_set

search_space = ps(
  C = p_dbl(lower = 0.1, upper = 0.5, depends = (R == FALSE)),
  M= p_int(lower = 1, upper = 10),
  N = p_int(lower = 2, upper = 10, depends = (R == TRUE)),
  R = p_lgl()
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_PART,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_PART$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939


#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

#     decay maxit size      rang       abstol       reltol learner_param_vals  x_domain classif.auc
# 0.0551435   255   10 0.5080007 6.807949e-05 5.769071e-05          <list[6]> <list[6]>   0.8695175

#         C M  N     R learner_param_vals  x_domain classif.auc
# 0.3599941 4 NA FALSE          <list[3]> <list[3]>   0.907499

## "classif.ranger"

learner_ranger = lrn("classif.ranger", predict_type = "prob")

learner_ranger$param_set

# Se escogen dos hiperparámetros y se configura el espacio de búsqueda

learner_ranger$param_set$values$splitrule = "extratrees"

search_space = ps(
  num.trees = p_int(lower = 1, upper = 1000),
  mtry = p_int(lower = 1, upper = 7),
  alpha = p_dbl(lower= 0, upper = 1),
  max.depth = p_int(lower = 0, upper = 100),
  min.node.size = p_int(lower = 1, upper = 100),
  minprop = p_dbl(lower= 0, upper = 0.5),
  num.random.splits = p_int(lower = 1, upper = 1000),
  num.threads = p_int(lower = 1, upper = 1000),
  sample.fraction = p_dbl(lower= 0.1, upper = 1),
  seed = p_int(lower = -1000, upper = 1000)
)


search_space

hout = rsmp("holdout")
measure = msr("classif.auc")

# Criterio de término

evals20 = trm("evals", n_evals = 100)

# Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_ranger,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

# Algoritmo de optimización

tuner = tnr("random_search")

# Gatillar el proceso de optimización

tuner$optimize(instance)

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939


#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

#     decay maxit size      rang       abstol       reltol learner_param_vals  x_domain classif.auc
# 0.0551435   255   10 0.5080007 6.807949e-05 5.769071e-05          <list[6]> <list[6]>   0.8695175

#         C M  N     R learner_param_vals  x_domain classif.auc
# 0.3599941 4 NA FALSE          <list[3]> <list[3]>   0.907499

# num.trees mtry     alpha max.depth min.node.size   minprop num.random.splits num.threads sample.fraction seed learner_param_vals
#       772    3 0.9425781        65             2 0.2011521                14         908       0.7666909 -961         <list[11]>

#    x_domain classif.auc
#1 <list[10]>   0.9447802

## "classif.rfsrc"

learner_rfsrc = mlr3::lrn("classif.rfsrc", predict_type = "prob")

learner_rfsrc$param_set$ids()

learner_rfsrc$param_set

search_space = ps(
  ntree = p_int(lower = 500, upper = 1500),
  mtry= p_int(lower = 3, upper = 7),
  nodesize = p_int(lower = 1, upper = 20),
  splitrule = p_fct(levels = c("gini", "auc", "entropy"))
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_rfsrc,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_rfsrc$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939


#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

#     decay maxit size      rang       abstol       reltol learner_param_vals  x_domain classif.auc
# 0.0551435   255   10 0.5080007 6.807949e-05 5.769071e-05          <list[6]> <list[6]>   0.8695175

#         C M  N     R learner_param_vals  x_domain classif.auc
# 0.3599941 4 NA FALSE          <list[3]> <list[3]>   0.907499

# num.trees mtry     alpha max.depth min.node.size   minprop num.random.splits num.threads sample.fraction seed learner_param_vals
#       772    3 0.9425781        65             2 0.2011521                14         908       0.7666909 -961         <list[11]>

#    x_domain classif.auc
#1 <list[10]>   0.9447802


# ntree mtry nodesize splitrule learner_param_vals  x_domain classif.auc
#  1472    5        2       auc          <list[4]> <list[4]>    0.939693

## "classif.rpart"

learner_rpart = mlr3::lrn("classif.rpart", predict_type = "prob")

learner_rpart$param_set$ids()

learner_rpart$param_set

search_space = ps(
  cp = p_dbl(lower = 0.001, upper = 0.1),
  maxdepth= p_int(lower = 5, upper = 30),
  minbucket = p_int(lower = 1, upper = 20),
  minsplit = p_int(lower = 10, upper = 50),
  xval = p_int(lower = 5, upper = 50)
)

search_space

### Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

### Criterio de término

evals20 = trm("evals", n_evals = 100)

### Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_rpart,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

### Algoritmo de optimización

tuner = tnr("random_search")

### Gatillar el proceso de optimización

set.seed(123)

tuner$optimize(instance)

### Valores de los hiperparámetros

instance$result_learner_param_vals

### Actualizar hiperparámetros

learner_rpart$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939


#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

#     decay maxit size      rang       abstol       reltol learner_param_vals  x_domain classif.auc
# 0.0551435   255   10 0.5080007 6.807949e-05 5.769071e-05          <list[6]> <list[6]>   0.8695175

#         C M  N     R learner_param_vals  x_domain classif.auc
# 0.3599941 4 NA FALSE          <list[3]> <list[3]>   0.907499

# num.trees mtry     alpha max.depth min.node.size   minprop num.random.splits num.threads sample.fraction seed learner_param_vals
#       772    3 0.9425781        65             2 0.2011521                14         908       0.7666909 -961         <list[11]>

#    x_domain classif.auc
#1 <list[10]>   0.9447802

# ntree mtry nodesize splitrule learner_param_vals  x_domain classif.auc
#  1472    5        2       auc          <list[4]> <list[4]>    0.939693

#           cp maxdepth minbucket minsplit xval learner_param_vals  x_domain classif.auc
#   0.01626603        7         3       38   33          <list[5]> <list[5]>   0.9071429

## "classif.svm"

learner_svm <- mlr3::lrn("classif.svm", type = "C-classification", kernel = "radial", predict_type = "prob")

learner_svm$param_set$ids()

set.seed(11187)
search_space = ps(
  cost = p_dbl(lower = 1000000, upper = 10000000),
  gamma = p_dbl(lower = 0.01, upper = 1),
  cachesize = p_dbl(lower = -10000, upper = 10000)
)


search_space

# Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

# Criterio de término

evals20 = trm("evals", n_evals = 100)

# Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_svm,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

# Algoritmo de optimización

tuner = tnr("random_search")

# Gatillar el proceso de optimización

tuner$optimize(instance)

# Valores de los hiperparámetros

instance$result_learner_param_vals

# Actualizar hiperparámetros

learner_svm$param_set$values = instance$result_learner_param_vals

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939


#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

#     decay maxit size      rang       abstol       reltol learner_param_vals  x_domain classif.auc
# 0.0551435   255   10 0.5080007 6.807949e-05 5.769071e-05          <list[6]> <list[6]>   0.8695175

#         C M  N     R learner_param_vals  x_domain classif.auc
# 0.3599941 4 NA FALSE          <list[3]> <list[3]>   0.907499

# num.trees mtry     alpha max.depth min.node.size   minprop num.random.splits num.threads sample.fraction seed learner_param_vals
#       772    3 0.9425781        65             2 0.2011521                14         908       0.7666909 -961         <list[11]>

#    x_domain classif.auc
#1 <list[10]>   0.9447802

# ntree mtry nodesize splitrule learner_param_vals  x_domain classif.auc
#  1472    5        2       auc          <list[4]> <list[4]>    0.939693

#           cp maxdepth minbucket minsplit xval learner_param_vals  x_domain classif.auc
#   0.01626603        7         3       38   33          <list[5]> <list[5]>   0.9071429

#    cost     gamma cachesize learner_param_vals  x_domain classif.auc
# 5202771 0.9625987 -2746.226          <list[5]> <list[3]>   0.9307168

## "classif.xgboost"

learner_xgboost = lrn("classif.xgboost", predict_type = "prob")

learner_xgboost$param_set

# Se escogen dos hiperparámetros y se configura el espacio de búsqueda

set.seed(1235)
search_space = ps(
  colsample_bytree = p_dbl(lower = 0.3, upper = 0.7),
  eta = p_dbl(lower = 0.2, upper = 0.8),
  gamma = p_dbl(lower = 0, upper = 10),
  max_depth = p_int(lower = 1, upper = 1000),
  min_child_weight = p_dbl(lower = 0, upper = 8),
  nrounds = p_int(lower = 1, upper = 100),
  subsample = p_dbl(lower = 0.5, upper = 1)
)


search_space

hout = rsmp("holdout")
measure = msr("classif.auc")

# Criterio de término

evals20 = trm("evals", n_evals = 100)

# Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_xgboost,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

# Algoritmo de optimización

tuner = tnr("random_search")

# Gatillar el proceso de optimización

tuner$optimize(instance)

# Valores de los hiperparámetros

instance$result_learner_param_vals

# Actualizar hiperparámetros

learner_xgboost$param_set$values = instance$result_learner_param_vals

#colsample_bytree      eta    gamma max_depth min_child_weight nrounds subsample learner_param_vals  x_domain classif.auc
#       0.4278874 0.382958 2.963799       588         2.107075      84 0.6077578         <list[10]> <list[7]>   0.9317725

###############################################

# Establecer los hiperparámetros para los learners

learner_adaboost = mlr3::lrn("classif.AdaBoostM1", predict_type = "prob", predict_sets = c("train", "test"))

learner_adaboost$param_set$values$I <- 67
learner_adaboost$param_set$values$P <- 96
learner_adaboost$param_set$values$S <- 6

# I  P S learner_param_vals  x_domain classif.auc
#67 96 6          <list[3]> <list[3]>   0.9592215

learner_bart = mlr3::lrn("classif.bart", predict_type = "prob")

learner_bart$param_set$values$ntree <- 114
learner_bart$param_set$values$k <- 0.4210805
learner_bart$param_set$values$power <- 1.093233
learner_bart$param_set$values$base <- 0.9273751
learner_bart$param_set$values$numcut <- 86

#ntree         k    power      base numcut learner_param_vals  x_domain classif.auc
#  114 0.4210805 1.093233 0.9273751     86          <list[5]> <list[5]>   0.9568713

learner_c50 = mlr3::lrn("classif.C50", predict_type = "prob")

learner_c50$param_set$values$trials <- 50
learner_c50$param_set$values$CF <- 0.253099
learner_c50$param_set$values$minCases <- 1

#trials       CF minCases learner_param_vals  x_domain classif.auc
#    50 0.253099        1          <list[3]> <list[3]>   0.9758551

learner_ctree = mlr3::lrn("classif.ctree", predict_type = "prob")

learner_ctree$param_set$values$testtype <- "MonteCarlo"

learner_ctree$param_set$values$minsplit <- 8
learner_ctree$param_set$values$minbucket <- 15
learner_ctree$param_set$values$maxdepth <- 9
learner_ctree$param_set$values$alpha <- 0.056352688
learner_ctree$param_set$values$mincriterion <- 0
learner_ctree$param_set$values$mtry <- 13
learner_ctree$param_set$values$maxsurrogate <- 2
learner_ctree$param_set$values$nresample <- 3594

#minsplit minbucket maxdepth      alpha mincriterion mtry maxsurrogate nresample   classif.auc
#       8        15        9 0.05635268            0   13            2      3594     0.9212234

learner_earth = mlr3::lrn("classif.earth", predict_type = "prob")

learner_earth$param_set$values$degree <- 2
learner_earth$param_set$values$penalty <- 2
learner_earth$param_set$values$thresh <- 0.004752231
learner_earth$param_set$values$minspan <- 18
learner_earth$param_set$values$endspan <- 7
learner_earth$param_set$values$nprune <- 19

#degree penalty      thresh minspan endspan nprune learner_param_vals  x_domain classif.auc
#     2       2 0.004752231      18       7     19          <list[6]> <list[6]>   0.9475895

learner_gam = mlr3::lrn("classif.gam", predict_type = "prob")

learner_gam$param_set$values$gamma <- 3.951286
learner_gam$param_set$values$epsilon <- 0.9545037
learner_gam$param_set$values$maxit <- 445
learner_gam$param_set$values$mgcv.tol <- 0.6928034

#   gamma   epsilon maxit  mgcv.tol learner_param_vals  x_domain classif.auc
#3.951286 0.9545037   445 0.6928034          <list[4]> <list[4]>   0.9287546

learner_gamboost = mlr3::lrn("classif.gamboost", predict_type = "prob")

learner_gamboost$param_set$values$dfbase <- 3
learner_gamboost$param_set$values$mstop <- 156
learner_gamboost$param_set$values$nu <- 0.01742525

#dfbase mstop         nu learner_param_vals  x_domain classif.auc
#     3   156 0.01742525          <list[3]> <list[3]>   0.9161797

learner_gausspr = mlr3::lrn("classif.gausspr", predict_type = "prob")

learner_gausspr$param_set$values$kernel <- "rbfdot"
learner_gausspr$param_set$values$sigma <- 1.153221
learner_gausspr$param_set$values$tol <- 0.9404732
learner_gausspr$param_set$values$fit <- TRUE

#kernel    sigma degree offset       tol  fit classif.auc warnings errors runtime_learners
#rbfdot 1.153221     NA     NA 0.9404732 TRUE   0.9565297        0      0             0.05

learner_glmboost = mlr3::lrn("classif.glmboost", predict_type = "prob")

learner_glmboost$param_set$values$mstop <- 198
learner_glmboost$param_set$values$nu <- 0.07018944
learner_glmboost$param_set$values$risk <- "oobag"

#mstop         nu  risk learner_param_vals  x_domain classif.auc
#  198 0.07018944 oobag          <list[3]> <list[3]>   0.9380822

learner_glmnet = mlr3::lrn("classif.glmnet", predict_type = "prob")

learner_glmnet$param_set$values$alpha <- 0.9798219
learner_glmnet$param_set$values$nlambda <- 55
learner_glmnet$param_set$values$standardize <- TRUE
learner_glmnet$param_set$values$intercept <- TRUE

#alpha    nlambda standardize intercept learner_param_vals  x_domain classif.auc
#0.9798219     55        TRUE      TRUE          <list[4]> <list[4]>   0.9438962

learner_IBk = mlr3::lrn("classif.IBk", predict_type = "prob")

learner_IBk$param_set$values$K <- 27
learner_IBk$param_set$values$W <- 0
learner_IBk$param_set$values$I <- TRUE

# K W    I learner_param_vals  x_domain classif.auc
#27 0 TRUE          <list[3]> <list[3]>    0.941798

learner_ksvm = learner_svm <- lrn("classif.ksvm", kernel = "rbfdot", predict_type = "prob", type = "C-svc", predict_sets = c("train", "test"))

learner_ksvm$param_set$values$sigma <- 0.7608749


# kernel     sigma degree learner_param_vals  x_domain classif.auc
# rbfdot 0.7608749     NA          <list[2]> <list[2]>   0.9498162

learner_lightgbm = mlr3::lrn("classif.lightgbm", predict_type = "prob")

learner_lightgbm$param_set$values$learning_rate <- 0.3667449
learner_lightgbm$param_set$values$num_leaves <- 179
learner_lightgbm$param_set$values$feature_fraction <- 0.33698
learner_lightgbm$param_set$values$bagging_fraction <- 0.67934
learner_lightgbm$param_set$values$bagging_freq <- 50
learner_lightgbm$param_set$values$max_depth <- 596
learner_lightgbm$param_set$values$min_data_in_leaf <- 31

#learning_rate num_leaves feature_fraction bagging_fraction bagging_freq max_depth min_data_in_leaf
#    0.3667449        179          0.33698          0.67934           50       596               31
#classif.auc
#0.9778672

learner_LMT = mlr3::lrn("classif.LMT", predict_type = "prob")


learner_LMT$param_set$values$C <- FALSE
learner_LMT$param_set$values$M <- 15
learner_LMT$param_set$values$W <- 0.04125137
learner_LMT$param_set$values$I <- 7
learner_LMT$param_set$values$P <- FALSE
learner_LMT$param_set$values$R <- TRUE

#  M          W I     P    R learner_param_vals  x_domain classif.auc
# 15 0.04125137 7 FALSE TRUE          <list[6]> <list[5]>    0.936038

learner_lr <- lrn("classif.log_reg", predict_type = "prob", predict_sets = c("train", "test"))

learner_lr$param_set$values$epsilon <- 2.876488e-07
learner_lr$param_set$values$maxit <- 81
learner_lr$param_set$values$trace <- TRUE

#     epsilon maxit trace learner_param_vals  x_domain classif.auc
#2.876488e-07    81  TRUE          <list[3]> <list[3]>   0.9237939

learner_naive_bayes = mlr3::lrn("classif.naive_bayes", predict_type = "prob")

learner_naive_bayes$param_set$values$eps <- 3.117091e-06
learner_naive_bayes$param_set$values$laplace <- 2.047375
learner_naive_bayes$param_set$values$threshold <- 0.002036244

#          eps  laplace   threshold learner_param_vals  x_domain classif.auc
# 3.117091e-06 2.047375 0.002036244          <list[3]> <list[3]>   0.8547149

learner_nnet = mlr3::lrn("classif.nnet", predict_type = "prob")

learner_nnet$param_set$values$decay <- 0.0551435
learner_nnet$param_set$values$maxit <- 255
learner_nnet$param_set$values$size <- 10
learner_nnet$param_set$values$rang <- 0.5080007
learner_nnet$param_set$values$abstol <- 6.807949e-05
learner_nnet$param_set$values$reltol <- 5.769071e-05

#     decay maxit size      rang       abstol       reltol learner_param_vals  x_domain classif.auc
# 0.0551435   255   10 0.5080007 6.807949e-05 5.769071e-05          <list[6]> <list[6]>   0.8695175

learner_PART = mlr3::lrn("classif.PART", predict_type = "prob")

learner_PART$param_set$values$C <- 0.3599941
learner_PART$param_set$values$M <- 4
learner_PART$param_set$values$R <- FALSE

#         C M  N     R learner_param_vals  x_domain classif.auc
# 0.3599941 4 NA FALSE          <list[3]> <list[3]>   0.907499

learner_rf = lrn("classif.ranger", predict_type = "prob", predict_sets = c("train", "test"))

learner_rf$param_set$values$splitrule <- "extratrees"

learner_rf$param_set$values$num.trees <- 250
learner_rf$param_set$values$mtry <- 2
learner_rf$param_set$values$alpha <- 1
learner_rf$param_set$values$max.depth <- 0
learner_rf$param_set$values$min.node.size <- 1
learner_rf$param_set$values$minprop <- 0.125
learner_rf$param_set$values$num.random.splits <- 1
learner_rf$param_set$values$num.threads <- 500
learner_rf$param_set$values$sample.fraction <- 0.7750
learner_rf$param_set$values$seed <- 1000

# num.trees mtry     alpha max.depth min.node.size   minprop num.random.splits num.threads sample.fraction seed learner_param_vals
#       772    3 0.9425781        65             2 0.2011521                14         908       0.7666909 -961         <list[11]>

#    x_domain classif.auc
#1 <list[10]>   0.9447802


learner_rfsrc = mlr3::lrn("classif.rfsrc", predict_type = "prob")

learner_rfsrc$param_set$values$ntree <- 1472
learner_rfsrc$param_set$values$mtry<- 5
learner_rfsrc$param_set$values$nodesize <- 2
learner_rfsrc$param_set$values$splitrule <- "auc"

# ntree mtry nodesize splitrule learner_param_vals  x_domain classif.auc
#  1472    5        2       auc          <list[4]> <list[4]>    0.939693

learner_rpart = mlr3::lrn("classif.rpart", predict_type = "prob")

learner_rpart$param_set$values$cp <- 0.01626603
learner_rpart$param_set$values$maxdepth <- 7
learner_rpart$param_set$values$minbucket <- 3
learner_rpart$param_set$values$minsplit <- 38
learner_rpart$param_set$values$xval <- 33

#           cp maxdepth minbucket minsplit xval learner_param_vals  x_domain classif.auc
#   0.01626603        7         3       38   33          <list[5]> <list[5]>   0.9071429

learner_svm <- mlr3::lrn("classif.svm", type = "C-classification", kernel = "radial", predict_type = "prob")

learner_svm$param_set$values$cost <- 5202771
learner_svm$param_set$values$gamma <- 0.9625987
learner_svm$param_set$values$cachesize <- -2746.226

#    cost     gamma cachesize learner_param_vals  x_domain classif.auc
# 5202771 0.9625987 -2746.226          <list[5]> <list[3]>   0.9307168

learner_xgboost <- lrn("classif.xgboost", predict_type = "prob", predict_sets = c("train", "test"))

learner_xgboost$param_set$values$colsample_bytree <- 0.5
learner_xgboost$param_set$values$eta <- 0.25
learner_xgboost$param_set$values$gamma <- 0
learner_xgboost$param_set$values$max_depth <- 250
learner_xgboost$param_set$values$min_child_weight <- 0
learner_xgboost$param_set$values$nrounds <- 75
learner_xgboost$param_set$values$subsample <- 0.625

#colsample_bytree      eta    gamma max_depth min_child_weight nrounds subsample learner_param_vals  x_domain classif.auc
#       0.4278874 0.382958 2.963799       588         2.107075      84 0.6077578         <list[10]> <list[7]>   0.9317725

set.seed(123456)

# Diseño del benchmark

design = benchmark_grid(
  tasks = task_v1,
  learners = list(learner_adaboost, learner_bart, learner_c50, learner_ctree, learner_earth, learner_gam, learner_gamboost, 
                  learner_gausspr, learner_glmboost, learner_glmnet, learner_IBk, learner_ksvm, learner_lightgbm, learner_LMT, 
                  learner_lr, learner_naive_bayes, learner_nnet, learner_PART, learner_rf, learner_rfsrc, learner_rpart, 
                  learner_svm, learner_xgboost),
  resamplings = rsmps("repeated_cv", folds = 5, repeats = 100)
)

print(design)

# Correr el benchmark

bmr = benchmark(design)

## Agregar las mediciones

measures = list(
  msr("classif.auc", predict_sets = "train", id = "auc_train"),
  msr("classif.auc", id = "auc_test", predict_sets = "test"),msr("classif.ce")
)

tab = bmr$aggregate(measures)

print(tab)

ranks = tab[, .(learner_id,rank_test = rank(-auc_test)), by = task_id]
print(ranks)

ranks[order(rank_test)]

# Test estadístico para medir diferencias (usando el error de clasificación)

resumen <- as.data.frame(bmr$score())

error <- resumen$classif.ce

modelos <- as.factor(c(rep(c("AB", "BART", "C50", "CTREE", "EARTH", "GAM", "GAMB", "GP", "GLMBO",
                                        "GLMNET","IBK","KSVM","LGBM","LMT","LR","NB","NNET","PART","RF","RFS","RPART",
                                        "SVM","XGB"), each =500)))

boxplot(error ~ modelos, col = c("blue", "yellow", "red","green"), ylab = "Classification Error", xlab = "Models")

estadistico <- as.data.frame(cbind(error,modelos))

estadistico<- mutate(estadistico,id = rep(1:500, 23))

friedman_test(estadistico, error ~ modelos | id)

friedman.test(estadistico$`resumen$classif.ce`, estadistico$modelos, estadistico$id)

#.y.       n statistic    df        p method
#* <chr> <int>     <dbl> <dbl>    <dbl> <chr>
#  error   500      100.     3 1.29e-21 Friedman test
# Hay diferencias significativas. Implica realizar un test Post-Hoc

resultados <- PMCMRplus::kwAllPairsNemenyiTest(error, modelos, method = "Chisq")

#       RF      RL      SVM
# RL  0.711   -       -
# SVM 0.245   0.018   -
# XGB 7.8e-09 3.4e-06 6.9e-14

matriz_p <- resultados$p.value

df_p <- as.data.frame(matriz_p)

df_p$Modelo1 <- rownames(df_p)

df_largo <- pivot_longer(df_p, cols = -Modelo1, names_to = "Modelo2", values_to = "p_valor")

df_final <- df_largo[!is.na(df_largo$p_valor) & df_largo$Modelo1 < df_largo$Modelo2, ]

df_final <- df_final[order(df_final$p_valor), ]

df_final$p_valor <- round(df_final$p_valor, 4)

tabla_latex <- xtable(df_final, caption = "Resultados del test de Nemenyi", label = "tab:nemenyi")

print(tabla_latex, include.rownames = FALSE)

library(ggplot2)
library(reshape2)

# Asumiendo que ya tienes la matriz de p-valores llamada 'matriz_p'

# Preparar los datos
matriz_p[upper.tri(matriz_p)] <- NA  # Eliminar el triángulo superior
diag(matriz_p) <- NA  # Eliminar la diagonal

# Convertir la matriz a formato largo
df_melt <- melt(matriz_p, na.rm = TRUE)

# Crear el mapa de calor
ggplot(df_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "blue", mid = "white", 
                       midpoint = 0.05, limit = c(0, 1), space = "Lab", 
                       name="p-valor") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(x = NULL, y = NULL, title = "Mapa de calor de p-valores del test de Nemenyi") +
  geom_text(aes(label = sprintf("%.3f", value)), color = "black", size = 3)

# Guardar el gráfico
ggsave("nemenyi_heatmap.pdf", width = 10, height = 8)

######################################################

learner_svm$param_set$values$C <- 888.8889
learner_svm$param_set$values$sigma <- 3.434409
learner_svm$param_set$values$cache <- 9091

#learner_xgboost$param_set$values$colsample_bytree <- 0.5
#learner_xgboost$param_set$values$eta <- 0.65
#learner_xgboost$param_set$values$gamma <- 5
#learner_xgboost$param_set$values$max_depth <- 250
#learner_xgboost$param_set$values$min_child_weight <- 2
#learner_xgboost$param_set$values$nrounds <- 100
#learner_xgboost$param_set$values$subsample <- 0.875

#learner_xgboost$param_set$values$colsample_bytree <- 0.6
#learner_xgboost$param_set$values$eta <- 0.2
#learner_xgboost$param_set$values$gamma <- 0
#learner_xgboost$param_set$values$max_depth <- 1000
#learner_xgboost$param_set$values$min_child_weight <- 0
#learner_xgboost$param_set$values$nrounds <- 100
#learner_xgboost$param_set$values$subsample <- 0.75

learner_xgboost$param_set$values$colsample_bytree <- 0.5
learner_xgboost$param_set$values$eta <- 0.25
learner_xgboost$param_set$values$gamma <- 0
learner_xgboost$param_set$values$max_depth <- 250
learner_xgboost$param_set$values$min_child_weight <- 0
learner_xgboost$param_set$values$nrounds <- 75
learner_xgboost$param_set$values$subsample <- 0.625

learner_rf$param_set$values$splitrule = "extratrees"

learner_rf$param_set$values$num.trees <- 250
learner_rf$param_set$values$mtry <- 2
learner_rf$param_set$values$alpha <- 1
learner_rf$param_set$values$max.depth <- 0
learner_rf$param_set$values$min.node.size <- 1
learner_rf$param_set$values$minprop <- 0.125
learner_rf$param_set$values$num.random.splits <- 1
learner_rf$param_set$values$num.threads <- 500
learner_rf$param_set$values$sample.fraction <- 0.7750
learner_rf$param_set$values$seed <- 1000

set.seed(123456)

# Diseño del benchmark

design = benchmark_grid(
  tasks = task_v1,
  learners = list(learner_svm,learner_lr,learner_rf,learner_xgboost),
  resamplings = rsmps("repeated_cv", repeats = 100, folds = 5)
)

print(design)

# Correr el benchmark

bmr = benchmark(design)

## Agregar las mediciones

measures = list(
  msr("classif.auc", predict_sets = "train", id = "auc_train"),
  msr("classif.auc", id = "auc_test", predict_sets = "test"),msr("classif.ce")
)

tab = bmr$aggregate(measures)

print(tab)

ranks = tab[, .(learner_id,rank_test = rank(-auc_test)), by = task_id]
print(ranks)

results <- bmr$score(measures = measures, predict_sets = "train")

results_svm <- results[results$learner_id == "classif.ksvm", ]

results_svm_grafico <- data.frame(
  AUC = c(results_svm$auc_train, results_svm$auc_test),
  Group = rep(c("Train", "Test"), each = nrow(results_svm))
)

ggplot(results_svm_grafico, aes(x = "Set", y = AUC, fill = Group)) +
  geom_boxplot() +
  scale_fill_manual(values = c("red", "blue")) +
  labs(x = "", y = "AUC",title = "(A) SVM") +
  theme_classic() + theme(plot.title = element_text(hjust = 0.5))

autoplot(bmr) + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

bmr_small = bmr$clone()$filter(task_id = "remociones")
autoplot(bmr_small, type = "roc")

###############################################################################3

# Test estadístico para medir diferencias (usando el error de clasificación)

resumen <- as.data.frame(bmr$score())

error <- resumen$classif.ce

modelos <- modelos <- as.factor(c(rep(c("SVM", "LR", "RF", "XGB"), each =500)))

boxplot(error ~ modelos, col = c("blue", "yellow", "red","green"), ylab = "Classification Error", xlab = "Models")

estadistico <- as.data.frame(cbind(error,modelos))

estadistico<- mutate(estadistico,id = rep(1:500, 4))

friedman_test(estadistico, error ~ modelos | id)

friedman.test(estadistico$`resumen$classif.ce`, estadistico$modelos, estadistico$id)

#.y.       n statistic    df        p method
#* <chr> <int>     <dbl> <dbl>    <dbl> <chr>
#  error   500      100.     3 1.29e-21 Friedman test
# Hay diferencias significativas. Implica realizar un test Post-Hoc

PMCMRplus::kwAllPairsNemenyiTest(error, modelos, method = "Chisq")

#       0RF      RL      SVM
# RL  0.711   -       -
# SVM 0.245   0.018   -
# XGB 7.8e-09 3.4e-06 6.9e-14

## Prueba de Kolgomorov-Smirnov para ver normalidad

ce_svm <- error[1:500]
ce_lr <- error[501:1000]
ce_rf <- error[1001:1500]
ce_xgboost <- error[1501:2000]

ks.test(ce_svm, pnorm)
ks.test(ce_lr , pnorm)
ks.test(ce_rf , pnorm)
ks.test(ce_xgboost , pnorm)

#Polígono de puntos y MAPA DE SUSCEPTIBILIDAD

taltal_poligono_sp <- shapefile("./La_negra_poligono.shp")

p1_taltal <- spsample(taltal_poligono_sp, n = 10000000, "regular")

p1_taltal_terra <- vect(p1_taltal)

#Carga de factores satelitales

EXG = 2 * B3 - B4 - B2
GRNDVI = (B5 - (B3 + B4))/(B5 + (B3 + B4))
MIRBI = 10.0 * B7 - 9.8 * B6 + 2.0
EVI = 2.5*((B5-B4)/(B5+6*B4-7.5*B2+1))
MLSWI26 = 	(1.0 - B5 - B6)/(1.0 - B5 + B6)

#Creación de pilas

pila_mapa_1 <-   c(tri,melton,hcurv,elevation) 

pila_mapa_2 <- c(edge_density,gaussian_curvature,spi)

pila_mapa_3 <- c(EXG,GRNDVI,MIRBI,EVI,MLSWI26)

rasValue_map <- terra::extract(pila_mapa_1, p1_taltal_terra)

rasValue1_map <- terra::extract(pila_mapa_2, p1_taltal_terra)

rasValue2_map <- terra::extract(pila_mapa_3, p1_taltal_terra)

valores_mapa <- cbind(as.data.frame(geom(p1_taltal_terra)),rasValue_map,rasValue1_map,rasValue2_map)

df_2 <- as.data.frame(valores_mapa)

# Limpiar df_2

df_2$ID <- NULL
df_2$ID <- NULL
df_2$ID <- NULL
df_2$geom <- NULL
df_2$part <- NULL
df_2$hole <- NULL

## Arreglar los nombres

names(df_2) <- c("x","y","tri","melton","hcurv","elevation","edge_density","gaussian_curvature","spi","exg","grndvi","mirbi","evi","mlswi26")

## Borrar los NAs

df_2 <- na.omit(df_2)

## Crear columna TARGET

df_2$REMOCION <- 1
df_2$REMOCION[1] <- 0

## Guardar a archivo

write.csv(df2,"./fabdem_satelitales/df2.csv")
write.csv(df_2,"./fabdem_satelitales/df_2.csv")

## Cargamos el learner con sus hiperparámetros respectivos

df_2 <- read.csv("./df_2.csv")
df2 <- read.csv("./df2.csv")


df_1$X <- NULL
df_1$id <- NULL

names(df_1)[1] <-"convergence_index"
names(df_1)[2] <- "tpi"
names(df_1)[3] <- "twi"
names(df_1)[4] <- "valley_depth"
names(df_1)[5] <- "evi"
names(df_1)[6] <- "ndgi"
names(df_1)[7] <- "plane_curvature"
names(df_1)[8] <- "x"
names(df_1)[9] <- "y"

df_2$REMOCION <- 1
df_2$REMOCION[1] <- 0

df_2$REMOCION <- as.factor(df_2$REMOCION)

df_2[] <- sapply(df_2, function(x) {x[is.infinite(x)] <- NA; return(x)})

df_2 <- na.omit(df_2)

task_v1 = mlr3spatiotempcv::TaskClassifST$new(
  id = "remociones",
  backend = df2,
  target = "REMOCION",
  coordinate_names = c("x", "y"),
  extra_args = list(
    coords_as_features = FALSE,
    crs = 4326)
)

task_df_2 = mlr3spatiotempcv::TaskClassifST$new(
  id = "remociones_1",
  backend = df_2,
  target = "REMOCION",
  coordinate_names = c("x", "y"),
  extra_args = list(
    coords_as_features = FALSE,
    crs = 4326)
)

## Usar en datos nuevos

learner_xgboost$train(task_v1)

prediccion <- predict(learner_xgboost, task_df_2$data(1:nrow(df_2)), predict_type = "prob")

pred <- as.data.frame(prediccion)

## Integrar al dataset generado ##

df_2$prediccion <- pred$`1`

## Crear archivo espacial de puntos ##

susceptibilidad <- data.frame(df_2$prediccion,df_2$x,df_2$y)

names(susceptibilidad)[1] <- "prediccion"
names(susceptibilidad)[2] <- "longitud"
names(susceptibilidad)[3] <- "latitud"

coordinates(susceptibilidad)= ~ longitud + latitud
proj4string(susceptibilidad) <- CRS("+init=epsg:4326")

susceptibilidad_terra<- vect(susceptibilidad)

terra::writeVector(susceptibilidad_terra,"./fabdem_satelitales/susceptibiidad_terra.shp")

## Transformar capa de puntos a Raster

r <- raster(ncol=3500, nrow=2800)
extent(r) <- extent(susceptibilidad)
rp <- rasterize(susceptibilidad, r, 'prediccion')

raster::writeRaster(rp,"./fabdem_satelitales/susceptibilidad_da_taltal.tif")

plot(susceptibilidad)

## Transformar capa de puntos a Raster

susceptibilidad_raster_lr <- terra::rasterize(susceptibilidad,salado, field="prediccion", updateValue="NA")

terra::writeRaster(susceptibilidad_raster_lr, filename = "./susceptibilidad_raster_lr.tif")

## MÉTODO JENK BREAKS

# Extraer los valores del raster
valores <- values(susceptibilidad_raster_lr)

# Definir el número de clases
n_clases <- 5

tamano_muestra <- 100000

# Tomar una muestra de los valores del raster
set.seed(123)  # Para reproducibilidad
indices_muestra <- sample(length(valores), tamano_muestra)
muestra <- valores[indices_muestra]

muestra <- muestra[!is.na(unlist(muestra))]

# Aplicar el método de Jenks Breaks a la muestra
n_clases <- 5
clases <- classInt::classIntervals(muestra, n = n_clases, style = "jenks")

## MÉTODO JENK BREAKS

# Extraer los valores del raster
valores <- values(slope)

# Definir el número de clases
n_clases <- 5

tamano_muestra <- 10000

# Tomar una muestra de los valores del raster
set.seed(123)  # Para reproducibilidad
indices_muestra <- sample(length(valores), tamano_muestra)
muestra <- valores[indices_muestra]

muestra <- muestra[!is.na(unlist(muestra))]

# Aplicar el método de Jenks Breaks a la muestra
n_clases <- 5
clases <- classInt::classIntervals(muestra, n = n_clases, style = "jenks")

################################
##AFINACIÓN DE HIPERPARÁMETROS##
################################

# Afinación de hiperparámetros


## XGBoost

learner = lrn("classif.xgboost", predict_type = "prob")
learner$param_set

# Se escogen dos hiperparámetros y se configura el espacio de búsqueda

set.seed(123456)
search_space = ps(
  colsample_bytree = p_dbl(lower = 0.3, upper = 0.7),
  eta = p_dbl(lower = 0.25, upper = 0.8),
  gamma = p_dbl(lower = 0, upper = 10),
  max_depth = p_int(lower = 1, upper = 1000),
  min_child_weight = p_dbl(lower = 0, upper = 8),
  nrounds = p_int(lower = 1, upper = 100),
  subsample = p_dbl(lower = 0.5, upper = 1)
)


search_space

hout = rsmp("holdout")
measure = msr("classif.auc")

# Criterio de término

evals20 = trm("perf_reached", level = 0.96)

# Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_xgboost,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

# Algoritmo de optimización

tuner = tnr("random_search")

# Gatillar el proceso de optimización

tuner$optimize(instance)

## Random Forest

## XGBoost

learner_ranger = lrn("classif.ranger", predict_type = "prob")
learner$param_set

# Se escogen dos hiperparámetros y se configura el espacio de búsqueda

learner_ranger$param_set$values$splitrule = "extratrees"

search_space = ps(
  num.trees = p_int(lower = 1, upper = 1000),
  mtry = p_int(lower = 1, upper = 7),
  alpha = p_dbl(lower= 0, upper = 1),
  max.depth = p_int(lower = 0, upper = 100),
  min.node.size = p_int(lower = 1, upper = 100),
  minprop = p_dbl(lower= 0, upper = 0.5),
  num.random.splits = p_int(lower = 1, upper = 1000),
  num.threads = p_int(lower = 1, upper = 1000),
  sample.fraction = p_dbl(lower= 0.1, upper = 1),
  seed = p_int(lower = -1000, upper = 1000)
)


search_space


search_space

hout = rsmp("holdout")
measure = msr("classif.auc")

# Criterio de término

evals20 = trm("evals", n_evals = 15000)

# Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_df2,
  learner = learner_ranger,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

# Algoritmo de optimización

tuner = tnr("grid_search", resolution = 5)

# Gatillar el proceso de optimización

tuner$optimize(instance)

# Usando Support Vector Machine

#learner_svm <- mlr3::lrn("classif.svm", type = "C-classification", kernel = "radial", predict_type = "prob")

learner_svm <- mlr3::lrn("classif.ksvm", predict_type = "prob", type = "C-svc", kernel = "rbfdot")

learner_svm$param_set$ids()

set.seed(123456)
search_space = ps(
  C = p_dbl(lower = 0.0001, upper = 1000),
  sigma = p_dbl(lower = 0.0001, upper = 10),
  cache = p_int(lower = 1, upper = 10000)
)


search_space

# Resampleo y métricas

hout = rsmp("holdout")
measure = msr("classif.auc")

# Criterio de término

evals20 = trm("perf_reached", level = 0.94)


# Creación de la instancia

instance = TuningInstanceSingleCrit$new(
  task = task_v1,
  learner = learner_svm,
  resampling = hout,
  measure = measure,
  search_space = search_space,
  terminator = evals20
)
instance

# Algoritmo de optimización

tuner = tnr("grid_search", resolution = 100)

# Gatillar el proceso de optimización

tuner$optimize(instance)

