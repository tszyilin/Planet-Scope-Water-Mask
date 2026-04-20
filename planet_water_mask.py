"""
Water Mask Analysis using Planet SDK (PlanetScope)
Equivalent to the Sentinel-2/GEE script, but using PlanetScope 3m imagery.

Supports: NDWI only (Green/NIR). NDPI is not possible — PlanetScope has no SWIR band.

Prerequisites:
    pip install planet rasterio numpy geopandas shapely pyproj folium nest_asyncio

Authentication:
    Set your API key below or as an environment variable:
        Windows: setx PL_API_KEY "your_key_here"
    Find your key at: https://www.planet.com/account/#/user-settings
"""

import os
import asyncio
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

# ── Suppress dependency warnings before any imports ───────────────────────────
# NumPy 2.0 copy warning (from rasterio/geopandas internals)
warnings.filterwarnings('ignore', message=".*copy while creating an array.*")
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Fix GDAL_DATA before rasterio/fiona load (prevents "Cannot find *.xsd/dxf" msgs)
if 'GDAL_DATA' not in os.environ:
    import sys
    # 1. Try Anaconda system GDAL
    _conda_gdal = Path(sys.prefix) / 'Library' / 'share' / 'gdal'
    # 2. Try rasterio bundled GDAL
    import rasterio as _rs
    _rasterio_gdal = Path(_rs.__file__).parent / 'gdal_data'
    for _candidate in [_conda_gdal, _rasterio_gdal]:
        if _candidate.exists():
            os.environ['GDAL_DATA'] = str(_candidate)
            break

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from pyproj import Transformer
import folium
import nest_asyncio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import planet
from planet import Auth, DataClient, OrdersClient, order_request

# nest_asyncio allows asyncio.run() inside Spyder's IPython kernel
nest_asyncio.apply()

# ─────────────────────────────────────────────
# INPUTS  (edit these)
# ─────────────────────────────────────────────

API_KEY = os.environ.get('PL_API_KEY', 'PLAK5d3461ccf4c74934bbba64a3b49f927e')

# ── Set this to skip search+order and go straight to download ──────────────
# Find your order ID at: https://www.planet.com/account/#/orders
# Set to None to run a fresh search and order
EXISTING_ORDER_ID = '7b0a00e3-d1c8-4b7a-82c8-02d12531a63d'

# Area of interest [xmin, ymin, xmax, ymax] in EPSG:7850
region_coords_7850 = [642899.883, 7044585.822, 644510.661, 7045912.346]

START_DATE   = '2023-04-05'
END_DATE     = '2023-04-30'
CLOUD_THRESH = 10       # max cloud cover %
NDWI_THRESH  = 0.1     # water = NDWI > threshold
MIN_AREA_M2  = 27      # drop polygons smaller than this (~3 pixels at 3 m)

OUTPUT_DIR = Path('planet_output')
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# GEOMETRY  (EPSG:7850 → WGS84 for Planet API)
# ─────────────────────────────────────────────

_tf = Transformer.from_crs('EPSG:7850', 'EPSG:4326', always_xy=True)
west,  south = _tf.transform(region_coords_7850[0], region_coords_7850[1])
east,  north = _tf.transform(region_coords_7850[2], region_coords_7850[3])

aoi_geojson = {
    "type": "Polygon",
    "coordinates": [[
        [west,  south],
        [east,  south],
        [east,  north],
        [west,  north],
        [west,  south],
    ]]
}

centroid_lon = (west  + east)  / 2
centroid_lat = (south + north) / 2

print(f'AOI (WGS84): W={west:.6f} S={south:.6f} E={east:.6f} N={north:.6f}')

# ─────────────────────────────────────────────
# STEP 1 — SEARCH
# ─────────────────────────────────────────────

async def search_scenes():
    """Search for PlanetScope scenes over the AOI."""
    auth = Auth.from_key(API_KEY)

    geom_filter  = planet.data_filter.geometry_filter(aoi_geojson)
    date_filter  = planet.data_filter.date_range_filter(
                       'acquired',
                       gte=datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc),
                       lte=datetime.fromisoformat(END_DATE).replace(tzinfo=timezone.utc))
    cloud_filter = planet.data_filter.range_filter(
                       'cloud_cover', lte=CLOUD_THRESH / 100)
    combined     = planet.data_filter.and_filter(
                       [geom_filter, date_filter, cloud_filter])

    items = []
    async with planet.Session(auth=auth) as sess:
        dc = DataClient(sess)
        async for item in dc.search(['PSScene'], combined, limit=100):
            assets = item['assets']
            if 'ortho_analytic_4b_sr' in assets or 'ortho_analytic_4b' in assets:
                items.append(item)

    print(f'\nFound {len(items)} PlanetScope scene(s):')
    for it in items:
        p = it['properties']
        print(f"  {it['id']}  acquired={p['acquired'][:10]}"
              f"  cloud={p['cloud_cover']*100:.1f}%")
    return items


# ─────────────────────────────────────────────
# STEP 2 — ORDER + DOWNLOAD
# ─────────────────────────────────────────────

async def order_and_download(item_ids: list[str]) -> Path:
    """Place a clipped order and download to OUTPUT_DIR."""
    auth = Auth.from_key(API_KEY)

    # Prefer surface reflectance bundle; analytic_sr_udm2 includes UDM2 cloud mask
    bundle = 'analytic_sr_udm2'

    request = order_request.build_request(
        name='planet_water_mask',
        products=[
            order_request.product(
                item_ids=item_ids,
                item_type='PSScene',
                product_bundle=bundle,
            )
        ],
        tools=[order_request.clip_tool(aoi_geojson)],
    )

    async with planet.Session(auth=auth) as sess:
        oc = OrdersClient(sess)
        order = await oc.create_order(request)
        order_id = order['id']
        print(f'\nOrder created: {order_id}')
        print('Waiting for Planet to process the order '
              '(usually 2–10 minutes)...')

        # Poll until complete (max_attempts=0 = unlimited, delay=30s between checks)
        await oc.wait(order_id, callback=lambda s: print(f'  state: {s}'),
                      delay=30, max_attempts=0)

        # Download
        await oc.download_order(order_id, directory=OUTPUT_DIR,
                                progress_bar=True)
        print(f'Downloaded → {OUTPUT_DIR}')
        return OUTPUT_DIR / order_id


# ─────────────────────────────────────────────
# STEP 3 — NDWI + WATER MASK (per scene)
# ─────────────────────────────────────────────

def compute_ndwi(tif_path: Path):
    """
    4-band PlanetScope SR band order: B1=Blue, B2=Green, B3=Red, B4=NIR
    NDWI (McFeeters) = (Green - NIR) / (Green + NIR)
    """
    with rasterio.open(tif_path) as src:
        green   = src.read(2).astype(np.float32)
        nir     = src.read(4).astype(np.float32)
        profile = src.profile.copy()
        nodata  = src.nodata

    if nodata is not None:
        valid = (green != nodata) & (nir != nodata)
    else:
        valid = np.ones(green.shape, dtype=bool)

    denom = green + nir
    ndwi  = np.where((denom != 0) & valid, (green - nir) / denom, np.nan)
    return ndwi, profile


def process_scene(tif_path: Path, date_str: str):
    print(f'\nProcessing {tif_path.name} ({date_str})')

    ndwi, profile = compute_ndwi(tif_path)

    valid = ~np.isnan(ndwi)
    print(f'  NDWI range: {np.nanmin(ndwi):.3f} → {np.nanmax(ndwi):.3f}')

    # ── Save NDWI raster ──────────────────────────────────────────────
    ndwi_path = OUTPUT_DIR / f'NDWI_{date_str}.tif'
    out_profile = profile.copy()
    out_profile.update(count=1, dtype='float32', nodata=np.nan)
    with rasterio.open(ndwi_path, 'w', **out_profile) as dst:
        dst.write(ndwi.astype('float32'), 1)
    print(f'  NDWI raster → {ndwi_path.name}')

    # ── Water mask ───────────────────────────────────────────────────
    water_mask = np.where(valid & (ndwi > NDWI_THRESH), 1, 0).astype(np.uint8)

    mask_path = OUTPUT_DIR / f'NDWIMask_{date_str}.tif'
    mask_profile = profile.copy()
    mask_profile.update(count=1, dtype='uint8', nodata=0)
    with rasterio.open(mask_path, 'w', **mask_profile) as dst:
        dst.write(water_mask, 1)
    print(f'  Mask raster  → {mask_path.name}')

    # ── Vectorise ────────────────────────────────────────────────────
    with rasterio.open(mask_path) as src:
        transform = src.transform
        crs       = src.crs

    polys = [
        shape(geom)
        for geom, val in shapes(water_mask, transform=transform)
        if val == 1
    ]

    if polys:
        gdf = gpd.GeoDataFrame({'date': date_str, 'geometry': polys}, crs=crs)
        gdf = gdf[gdf.geometry.area > MIN_AREA_M2].reset_index(drop=True)
    else:
        gdf = gpd.GeoDataFrame(columns=['date', 'geometry'], crs=crs)

    print(f'  Water polygons: {len(gdf)}')
    return gdf


# ─────────────────────────────────────────────
# STEP 4 — EXPORT + MAP
# ─────────────────────────────────────────────

def export_shapefile(gdfs: dict[str, gpd.GeoDataFrame]):
    """Merge all dates and export as SHP in EPSG:7850."""
    frames = [gdf.to_crs('EPSG:7850') for gdf in gdfs.values() if not gdf.empty]
    if not frames:
        print('No water polygons to export.')
        return
    combined = gpd.pd.concat(frames, ignore_index=True)
    shp_path = OUTPUT_DIR / 'NDWIMask_water_planet.shp'
    combined.to_file(shp_path)
    print(f'\nShapefile exported → {shp_path}')


def plot_ndwi_histograms(ndwi_arrays: dict[str, np.ndarray]):
    """Plot NDWI histogram for each scene, with water threshold marked."""
    n = len(ndwi_arrays)
    if n == 0:
        print('No NDWI data to plot.')
        return

    fig, axes = plt.subplots(n, 1, figsize=(9, 4 * n), squeeze=False)
    fig.suptitle('NDWI Histograms — PlanetScope', fontsize=14, fontweight='bold', y=1.01)

    for ax, (date_str, ndwi) in zip(axes[:, 0], ndwi_arrays.items()):
        valid = ndwi[~np.isnan(ndwi)].ravel()

        # Histogram
        counts, bins, patches = ax.hist(
            valid, bins=100, range=(-1, 1),
            color='steelblue', edgecolor='none', alpha=0.85
        )

        # Colour bars above threshold in blue, below in tan
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor('#2171b5' if left >= NDWI_THRESH else '#d4b483')

        # Threshold line
        ax.axvline(NDWI_THRESH, color='red', linewidth=1.5,
                   linestyle='--', label=f'Threshold = {NDWI_THRESH}')

        # Water fraction annotation
        water_pct = (valid >= NDWI_THRESH).sum() / len(valid) * 100
        ax.text(0.98, 0.95, f'Water pixels: {water_pct:.1f}%',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, color='#2171b5',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        ax.set_title(f'Date: {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}',
                     fontsize=11)
        ax.set_xlabel('NDWI')
        ax.set_ylabel('Pixel count')
        ax.set_xlim(-1, 1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{int(x):,}'))
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    hist_path = OUTPUT_DIR / 'NDWI_histograms.png'
    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f'Histogram saved → {hist_path}')
    plt.show()


def make_map(gdfs_wgs84: dict[str, gpd.GeoDataFrame]):
    fmap = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=15)

    # AOI outline
    folium.GeoJson(
        aoi_geojson, name='AOI',
        style_function=lambda _: {'color': 'red', 'fill': False, 'weight': 2}
    ).add_to(fmap)

    colors = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']
    for i, (date_str, gdf) in enumerate(gdfs_wgs84.items()):
        if not gdf.empty:
            c = colors[i % len(colors)]
            folium.GeoJson(
                gdf.__geo_interface__,
                name=f'Water {date_str}',
                style_function=lambda _, col=c: {
                    'fillColor': col, 'color': col,
                    'fillOpacity': 0.55, 'weight': 1
                }
            ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    map_path = OUTPUT_DIR / 'planet_water_mask_map.html'
    fmap.save(str(map_path))
    print(f'Map saved → {map_path}  (open in browser)')


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

async def download_existing_order(order_id: str):
    """Wait for an order to finish then download it."""
    auth = Auth.from_key(API_KEY)
    async with planet.Session(auth=auth) as sess:
        oc = OrdersClient(sess)
        print(f'Waiting for order {order_id} to finish (checking every 30s)...')
        await oc.wait(order_id, callback=lambda s: print(f'  state: {s}'),
                      delay=30, max_attempts=0)
        print('Order complete. Downloading...')
        await oc.download_order(order_id, directory=OUTPUT_DIR, progress_bar=True)
        print(f'Downloaded → {OUTPUT_DIR}')


async def main():
    if EXISTING_ORDER_ID:
        # ── Skip search+order, download directly ─────────────────────
        await download_existing_order(EXISTING_ORDER_ID)
    else:
        # ── Search ───────────────────────────────────────────────────
        items = await search_scenes()
        if not items:
            print('No scenes found. Adjust date range or cloud threshold.')
            return
        item_ids = [it['id'] for it in items]

        # ── Order & download ──────────────────────────────────────────
        await order_and_download(item_ids)

    # ── Find downloaded TIFs (SR files contain "_SR_" in filename) ────
    tif_files = sorted(OUTPUT_DIR.rglob('*_SR_*.tif'))
    if not tif_files:                          # fallback: any tif
        tif_files = sorted(OUTPUT_DIR.rglob('*.tif'))
    print(f'\nTIF files to process: {len(tif_files)}')

    # ── Process each scene ────────────────────────────────────────────
    gdfs_native  = {}
    ndwi_arrays  = {}
    for tif in tif_files:
        # Planet filename starts with date: 20230410_...
        date_str = tif.stem[:8] if tif.stem[:8].isdigit() else tif.stem
        gdf = process_scene(tif, date_str)
        gdfs_native[date_str] = gdf

        # Load the saved NDWI raster for plotting
        ndwi_path = OUTPUT_DIR / f'NDWI_{date_str}.tif'
        if ndwi_path.exists():
            with rasterio.open(ndwi_path) as src:
                ndwi_arrays[date_str] = src.read(1).astype(np.float32)

    # ── Histogram ─────────────────────────────────────────────────────
    plot_ndwi_histograms(ndwi_arrays)

    # ── Export ────────────────────────────────────────────────────────
    export_shapefile(gdfs_native)

    gdfs_wgs84 = {d: g.to_crs('EPSG:4326')
                  for d, g in gdfs_native.items() if not g.empty}
    make_map(gdfs_wgs84)


if __name__ == '__main__':
    asyncio.run(main())
