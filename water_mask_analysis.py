"""
Water Mask Analysis using Google Earth Engine Python API + folium
Converted from GEE JavaScript Code Editor script.

Prerequisites:
  pip install earthengine-api folium

Authentication (first time only):
  Run in Spyder console: import ee; ee.Authenticate(); ee.Initialize()
  Or in terminal:        earthengine authenticate
"""

import ee
import folium

# Helper: add a GEE layer to a folium map
def add_ee_layer(fmap, ee_object, vis_params, name):
    map_id_dict = ee.Image(ee_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(fmap)

# ─────────────────────────────────────────────
# INPUTS
# ─────────────────────────────────────────────

# Area of interest in EPSG:7850
region_coords = [642899.883, 7044585.822, 644510.661, 7045912.346]  # [xmin, ymin, xmax, ymax]

# Sentinel-2 tile
tile = '50JPR'

# Index to use for the mask
indexmask = 'NDWIMask'   # 'NDWIMask' or 'NDPIMask'

# Thresholds
ndwi_thresh = 0.1
ndpi_thresh = 0.0

# Date range
StartDate = '2023-04-05'
EndDate   = '2023-04-30'

# Export folder on Google Drive
export_folder = 'Bassetts'

# ─────────────────────────────────────────────
# INITIALISE
# ─────────────────────────────────────────────
ee.Initialize(project='psm6160-bassetts')  # e.g. 'my-gee-project-123'

# ─────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────
region = ee.Geometry.Rectangle(
    coords=region_coords,
    proj='EPSG:7850',
    evenOdd=False
)

# Reproject to WGS84 for GEE compatibility
aoi = region.transform('EPSG:4326', 1)
print('AOI (WGS84):', aoi.getInfo())

aoiCentroid = aoi.centroid(maxError=1)
coords = aoiCentroid.coordinates().getInfo()
east, north = coords[0], coords[1]
print(f'Centroid — lon: {east:.6f}, lat: {north:.6f}')

# ─────────────────────────────────────────────
# BAND INDEX FUNCTIONS
# ─────────────────────────────────────────────

def addNDWI(image):
    """NDWI (McFeeters): (B3 - B8A) / (B3 + B8A)"""
    ndwi = image.normalizedDifference(['B3', 'B8A']).rename('NDWI')
    return image.addBands(ndwi)

def addNDWI_Mask(image):
    mask = image.select('NDWI').gt(ndwi_thresh).rename('NDWIMask')
    return image.addBands(mask)

def addNDPI(image):
    """NDPI: (B12 - B3) / (B12 + B3)"""
    ndpi = image.normalizedDifference(['B12', 'B3']).rename('NDPI')
    return image.addBands(ndpi)

def addNDPI_Mask(image):
    mask = image.select('NDPI').lt(ndpi_thresh).rename('NDPIMask')
    return image.addBands(mask)

def maskS2clouds(image):
    """Mask clouds using QA60 band."""
    qa = image.select('QA60')
    cloudBitMask  = 1 << 10
    cirrusBitMask = 1 << 11
    mask = (qa.bitwiseAnd(cloudBitMask).eq(0)
              .And(qa.bitwiseAnd(cirrusBitMask).eq(0)))
    return image.updateMask(mask).divide(10000)

def get_cloud_cover_roi(image):
    """Estimate cloud cover percentage within AOI."""
    Img_to_pixelscount = (image.select('NDWI')
        .reduceRegion(reducer=ee.Reducer.count(), geometry=image.geometry(),
                      scale=10, maxPixels=1e9)
        .get('NDWI'))
    npix = (image.select('NDWI').unmask()
        .reduceRegion(reducer=ee.Reducer.count(), geometry=image.geometry(),
                      scale=10, maxPixels=1e9)
        .get('NDWI'))
    cloud_cover_roi = (ee.Number(1)
        .subtract(ee.Number(Img_to_pixelscount).divide(npix))
        .multiply(100))
    return image.set('cloud_cover_roi', cloud_cover_roi)

def find_date(img):
    idate = ee.String(ee.Image(img).get('system:index'))
    return ee.Date.parse('yyyyMMdd', idate.slice(0, 8)).millis()

def addDate(img):
    return img.set({'start_date': find_date(img)})

# ─────────────────────────────────────────────
# BUILD IMAGE COLLECTION
# ─────────────────────────────────────────────
dataset = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
    .filterDate(StartDate, EndDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 2))
    .filter(ee.Filter.eq('MGRS_TILE', tile))
    .map(maskS2clouds)
    .map(lambda img: img.clip(aoi))
    .map(addNDWI)
    .map(addNDWI_Mask)
    .map(addNDPI)
    .map(addNDPI_Mask))

dataset_filtered = (dataset
    .map(get_cloud_cover_roi)
    .filter(ee.Filter.lt('cloud_cover_roi', 10)))

print('Original dataset count:', dataset.size().getInfo())
print('Filtered dataset count:', dataset_filtered.size().getInfo())

# ─────────────────────────────────────────────
# IMAGE DATES
# ─────────────────────────────────────────────
S2dates = dataset_filtered.toList(dataset.size()).map(find_date)
readable_dates = S2dates.map(lambda millis: ee.Date(millis).format()).getInfo()
print('Available dates:', readable_dates)

# ─────────────────────────────────────────────
# COLLECTION FOR ANALYSIS
# ─────────────────────────────────────────────
collection = (dataset_filtered
    .select(['NDWI', 'NDWIMask', 'NDPI', 'NDPIMask'])
    .map(addDate))

# ─────────────────────────────────────────────
# MAP DISPLAY
# ─────────────────────────────────────────────
map_image = dataset_filtered.select(indexmask).sum()

fmap = folium.Map(location=[north, east], zoom_start=14)

add_ee_layer(fmap, dataset_filtered.median(),
             {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3},
             'True Colour (median)')

add_ee_layer(fmap, map_image,
             {'bands': [indexmask],
              'palette': ['ffffff', '9ecae1', '3182bd', '08519c'],
              'min': 0, 'max': dataset_filtered.size().getInfo()},
             f'{indexmask} Sum')

folium.LayerControl().add_to(fmap)

map_path = 'water_mask_map.html'
fmap.save(map_path)
print(f'Map saved → {map_path}  (open in your browser)')

# ─────────────────────────────────────────────
# VECTOR CONVERSION (first image only — preview)
# ─────────────────────────────────────────────
first_image = collection.select(indexmask).first()

vector = (first_image.reduceToVectors(
    scale=10,
    geometryType='polygon',
    labelProperty='water',
    geometryInNativeProjection=True
).filter(ee.Filter.eq('water', 1))
 .filter(ee.Filter.gt('count', 2)))

print('Vector polygon count (first image):', vector.size().getInfo())

# ─────────────────────────────────────────────
# POLYGON EXPORT (all dates → Google Drive)
# ─────────────────────────────────────────────

def convertPolygon(image):
    fc = image.reduceToVectors(
        scale=10,
        geometryType='polygon',
        labelProperty='water',
        geometryInNativeProjection=True
    )
    date_str = ee.Date(image.get('start_date')).format('YYYY-MM-dd')
    fc_lake = (fc.filter(ee.Filter.eq('water', 1))
                 .filter(ee.Filter.gt('count', 2))
                 .map(lambda feat: feat.set('date', date_str)))
    fc_land = (fc.filter(ee.Filter.eq('water', 1))
                 .filter(ee.Filter.gt('count', 10))
                 .map(lambda feat: feat.set('date', date_str)))
    return ee.Algorithms.If(fc_lake.size(), fc_lake, fc_land)

mask_poly      = collection.select(indexmask).map(convertPolygon)
mask_poly_flat = ee.FeatureCollection(mask_poly).flatten()

# Export water mask polygons
task_shp = ee.batch.Export.table.toDrive(
    collection=mask_poly_flat,
    description=f'Mask_water_{indexmask}',
    folder=export_folder,
    fileFormat='SHP'
)
task_shp.start()
print(f'Shapefile export task started: Mask_water_{indexmask}')

# Export index sum raster
task_tif = ee.batch.Export.image.toDrive(
    image=map_image.toInt(),
    description=indexmask,
    folder=export_folder,
    region=aoi,
    scale=10,
    crs='EPSG:7850'
)
task_tif.start()
print(f'GeoTIFF export task started: {indexmask}')

print('\nCheck export progress at: https://code.earthengine.google.com/tasks')
