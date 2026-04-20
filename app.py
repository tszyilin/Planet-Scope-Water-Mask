"""
Planet Water Mask Analyser — Streamlit Web App

Run with:
    streamlit run app.py

Install dependencies:
    pip install streamlit streamlit-folium planet rasterio numpy geopandas
                shapely pyproj folium nest_asyncio matplotlib scikit-image
"""

# ── GDAL / NumPy warnings suppressed before any rasterio/geopandas import ──
import os, sys, warnings
if 'GDAL_DATA' not in os.environ:
    import rasterio as _rs
    _conda = os.path.join(sys.prefix, 'Library', 'share', 'gdal')
    _rio   = os.path.join(os.path.dirname(_rs.__file__), 'gdal_data')
    for _c in [_conda, _rio]:
        if os.path.isdir(_c):
            os.environ['GDAL_DATA'] = _c
            break
warnings.filterwarnings('ignore', message='.*copy while creating an array.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*Memory.*driver is deprecated.*")

import asyncio, io, zipfile, tempfile
from datetime import datetime, timezone, date
from pathlib import Path

# Detect whether we are running on Streamlit Cloud (no display server)
_IS_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') == '1' \
            or os.environ.get('IS_STREAMLIT_CLOUD') == 'true' \
            or not os.environ.get('DISPLAY', True)  # True = local fallback

try:
    import tkinter as tk
    from tkinter import filedialog
    _HAS_TKINTER = True
except Exception:
    _HAS_TKINTER = False

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from pyproj import Transformer
import folium
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from skimage.filters import threshold_otsu
import nest_asyncio
import streamlit as st
from streamlit_folium import st_folium

import planet
from planet import Auth, DataClient, OrdersClient, order_request

nest_asyncio.apply()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title='Planet Water Mask Analyser',
    page_icon='💧',
    layout='wide',
)

def browse_folder():
    """Open a native folder picker — local only, disabled on Streamlit Cloud."""
    if not _HAS_TKINTER:
        return None
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory(title='Select output folder')
    root.destroy()
    return folder

# Temp dir used on cloud so all processing still works without local paths
_TEMP_DIR = Path(tempfile.mkdtemp())

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
_defaults = dict(
    download_done=False,
    process_done=False,
    tif_files=[],
    ndwi_arrays={},
    otsu_values={},
    gdfs_wgs84={},
    zip_shp_bytes=None,
    zip_tif_bytes=None,
    aoi_geojson=None,
    centroid_lat=None,
    centroid_lon=None,
    dir_download=str(_TEMP_DIR / 'download'),
    dir_ndwi=str(_TEMP_DIR / 'ndwi'),
    dir_shp=str(_TEMP_DIR / 'shapefiles'),
)
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title('💧 Water Mask Analyser')
    st.caption('PlanetScope · NDWI · Streamlit')

    # API key: use Streamlit Secrets on cloud, editable field locally
    _secret_key = st.secrets.get('PLANET_API_KEY', '') if hasattr(st, 'secrets') else ''
    api_key = st.text_input('Planet API Key', type='password',
                             value=_secret_key or 'PLAK5d3461ccf4c74934bbba64a3b49f927e',
                             help='On Streamlit Cloud set PLANET_API_KEY in Secrets.')

    st.divider()
    order_mode = st.radio('Data source',
                          ['Use existing order', 'Search & order new data'])

    # ── EPSG selector (shared by both modes) ─────────────────────────
    st.subheader('Coordinate System')
    epsg_input = st.text_input('EPSG code', value='7850',
                               help='Enter the EPSG code for your AOI coordinates. '
                                    'e.g. 7850 (MGA2020 Z50), 28350 (GDA94 Z50), '
                                    '32755 (WGS84 UTM Z55S), 4326 (WGS84 lat/lon)')
    # Validate EPSG live
    epsg_valid = False
    epsg_error = ''
    try:
        from pyproj import CRS
        _crs_obj  = CRS.from_epsg(int(epsg_input))
        epsg_valid = True
        epsg_label = _crs_obj.name
        st.caption(f'✅ {epsg_label}')
    except Exception:
        epsg_error = f'EPSG:{epsg_input} not recognised — check the code.'
        st.error(epsg_error)

    # Coordinate labels change for geographic vs projected CRS
    is_geographic = epsg_valid and _crs_obj.is_geographic
    if is_geographic:
        x_label, y_label = 'Lon min', 'Lon max'
        x2_label, y2_label = 'Lat min', 'Lat max'
        fmt = '%.6f'
        default_vals = (115.0, -32.0, 116.0, -31.0)   # generic WGS84 fallback
    else:
        x_label, y_label = 'X min (Easting)', 'X max (Easting)'
        x2_label, y2_label = 'Y min (Northing)', 'Y max (Northing)'
        fmt = '%.3f'
        default_vals = (642899.883, 7044585.822, 644510.661, 7045912.346)

    if order_mode == 'Use existing order':
        existing_order_id = st.text_input(
            'Order ID', value='7b0a00e3-d1c8-4b7a-82c8-02d12531a63d',
            placeholder='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx')
        aoi_xmin, aoi_ymin = default_vals[0], default_vals[1]
        aoi_xmax, aoi_ymax = default_vals[2], default_vals[3]
        start_date_val = date(2023, 4, 5)
        end_date_val   = date(2023, 4, 30)
        cloud_pct      = 10
    else:
        existing_order_id = None
        st.subheader('AOI Extent')
        c1, c2 = st.columns(2)
        aoi_xmin = c1.number_input(x_label,  value=default_vals[0], format=fmt)
        aoi_ymin = c1.number_input(x2_label, value=default_vals[1], format=fmt)
        aoi_xmax = c2.number_input(y_label,  value=default_vals[2], format=fmt)
        aoi_ymax = c2.number_input(y2_label, value=default_vals[3], format=fmt)
        st.subheader('Date range')
        start_date_val = st.date_input('Start', value=date(2023, 4, 5))
        end_date_val   = st.date_input('End',   value=date(2023, 4, 30))
        cloud_pct      = st.slider('Max cloud cover (%)', 0, 100, 10)

    st.divider()
    if _HAS_TKINTER:
        st.subheader('Output Folders')
        st.caption('Files also available via download buttons in the Results tab.')

        def _folder_row(label, state_key, browse_key):
            fc1, fc2 = st.columns([3, 1])
            val = fc1.text_input(label, value=st.session_state[state_key],
                                 key=f'_input_{state_key}')
            if fc2.button('Browse', key=browse_key, use_container_width=True):
                picked = browse_folder()
                if picked:
                    st.session_state[state_key] = picked
                    st.rerun()
            st.session_state[state_key] = val

        _folder_row('Downloaded imagery',  'dir_download', 'browse_dl')
        _folder_row('NDWI rasters',        'dir_ndwi',     'browse_ndwi')
        _folder_row('Shapefiles',          'dir_shp',      'browse_shp')
    else:
        st.info('☁️ Running on Streamlit Cloud — outputs available via download buttons in the Results tab.')

    st.divider()
    min_area_m2 = st.number_input('Min polygon area (m²)', value=27, min_value=1)

# Derive AOI geometry (reproject to WGS84 regardless of input EPSG)
_src_epsg = f'EPSG:{epsg_input}' if epsg_valid else 'EPSG:7850'
_tf = Transformer.from_crs(_src_epsg, 'EPSG:4326', always_xy=True)
w, s = _tf.transform(aoi_xmin, aoi_ymin)
e, n = _tf.transform(aoi_xmax, aoi_ymax)
# Ensure correct min/max ordering after reprojection
w, e = min(w, e), max(w, e)
s, n = min(s, n), max(s, n)
aoi_geojson = {'type': 'Polygon', 'coordinates': [[[w,s],[e,s],[e,n],[w,n],[w,s]]]}
st.session_state.aoi_geojson  = aoi_geojson
st.session_state.centroid_lat = (s + n) / 2
st.session_state.centroid_lon = (w + e) / 2

# Resolve all three output directories and ensure they exist
DIR_DOWNLOAD = Path(st.session_state.dir_download)
DIR_NDWI     = Path(st.session_state.dir_ndwi)
DIR_SHP      = Path(st.session_state.dir_shp)
for _d in [DIR_DOWNLOAD, DIR_NDWI, DIR_SHP]:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception as _e:
        st.sidebar.error(f'Cannot create folder {_d}: {_e}')

# ─────────────────────────────────────────────
# ASYNC HELPERS
# ─────────────────────────────────────────────
def run(coro):
    return asyncio.run(coro)

async def _search(api_key, aoi_geojson, start_date_val, end_date_val, cloud_pct):
    auth = Auth.from_key(api_key)
    gf = planet.data_filter.geometry_filter(aoi_geojson)
    df = planet.data_filter.date_range_filter(
        'acquired',
        gte=datetime.combine(start_date_val, datetime.min.time()).replace(tzinfo=timezone.utc),
        lte=datetime.combine(end_date_val,   datetime.max.time()).replace(tzinfo=timezone.utc))
    cf = planet.data_filter.range_filter('cloud_cover', lte=cloud_pct / 100)
    combined = planet.data_filter.and_filter([gf, df, cf])
    items = []
    async with planet.Session(auth=auth) as sess:
        dc = DataClient(sess)
        async for item in dc.search(['PSScene'], combined, limit=100):
            if ('ortho_analytic_4b_sr' in item['assets'] or
                    'ortho_analytic_4b' in item['assets']):
                items.append(item)
    return items

async def _order_and_download(api_key, item_ids, aoi_geojson, status_cb):
    auth = Auth.from_key(api_key)
    req = order_request.build_request(
        name='streamlit_water_mask',
        products=[order_request.product(
            item_ids=item_ids, item_type='PSScene',
            product_bundle='analytic_sr_udm2')],
        tools=[order_request.clip_tool(aoi_geojson)],
    )
    async with planet.Session(auth=auth) as sess:
        oc = OrdersClient(sess)
        order = await oc.create_order(req)
        order_id = order['id']
        status_cb(f'Order created: `{order_id}` — waiting for Planet to process...')
        await oc.wait(order_id,
                      callback=lambda s: status_cb(f'Planet order state: **{s}**'),
                      delay=30, max_attempts=0)
        await oc.download_order(order_id, directory=DIR_DOWNLOAD, progress_bar=False)
    return order_id

async def _download_existing(api_key, order_id, status_cb):
    auth = Auth.from_key(api_key)
    async with planet.Session(auth=auth) as sess:
        oc = OrdersClient(sess)
        status_cb(f'Waiting for order `{order_id}`...')
        await oc.wait(order_id,
                      callback=lambda s: status_cb(f'Planet order state: **{s}**'),
                      delay=30, max_attempts=0)
        await oc.download_order(order_id, directory=DIR_DOWNLOAD, progress_bar=False)

# ─────────────────────────────────────────────
# PROCESSING HELPERS
# ─────────────────────────────────────────────
def compute_ndwi(tif_path):
    with rasterio.open(tif_path) as src:
        if src.count < 4:
            raise ValueError(
                f'{tif_path.name} has only {src.count} band(s) — '
                f'NIR (band 4) is required for NDWI. '
                f'This is likely a 3-band (BGR) product with no NIR. '
                f'Re-order using the "analytic_sr_udm2" bundle which includes NIR.')
        green   = src.read(2).astype(np.float32)
        nir     = src.read(4).astype(np.float32)
        profile = src.profile.copy()
        nodata  = src.nodata
    valid = (green != nodata) & (nir != nodata) if nodata is not None \
            else np.ones(green.shape, bool)
    denom = green + nir
    with np.errstate(invalid='ignore', divide='ignore'):
        ndwi = np.where((denom != 0) & valid, (green - nir) / denom, np.nan)
    return ndwi, profile

def otsu_threshold(ndwi):
    flat  = ndwi.ravel()
    valid = flat[np.isfinite(flat) & (flat >= -1) & (flat <= 1)]
    if valid.size < 2:
        return 0.1
    return float(threshold_otsu(valid))

def process_scene(tif_path, date_str, threshold, ndwi_dir, shp_dir):
    ndwi, profile = compute_ndwi(tif_path)

    # Save NDWI raster
    ndwi_path = ndwi_dir / f'NDWI_{date_str}.tif'
    p = profile.copy(); p.update(count=1, dtype='float32', nodata=np.nan)
    with rasterio.open(ndwi_path, 'w', **p) as dst:
        dst.write(ndwi.astype('float32'), 1)

    # Water mask
    water = np.where(np.isfinite(ndwi) & (ndwi > threshold), 1, 0).astype(np.uint8)
    mask_path = ndwi_dir / f'NDWIMask_{date_str}.tif'
    p2 = profile.copy(); p2.update(count=1, dtype='uint8', nodata=0)
    with rasterio.open(mask_path, 'w', **p2) as dst:
        dst.write(water, 1)

    # Vectorise
    with rasterio.open(mask_path) as src:
        tf2, crs = src.transform, src.crs
    polys = [shape(g) for g, v in shapes(water, transform=tf2) if v == 1]
    if polys:
        gdf = gpd.GeoDataFrame({'date': date_str, 'geometry': polys}, crs=crs)
        gdf = gdf[gdf.geometry.area > min_area_m2].reset_index(drop=True)
    else:
        gdf = gpd.GeoDataFrame(columns=['date','geometry'], crs=crs)

    return ndwi, gdf

def build_histogram(date_str, ndwi, manual_thresh, otsu_thresh=None):
    valid = ndwi[np.isfinite(ndwi)].ravel()
    active_thresh = otsu_thresh if otsu_thresh is not None else manual_thresh

    fig, ax = plt.subplots(figsize=(9, 3.5))
    counts, bins, patches = ax.hist(valid, bins=120, range=(-1, 1),
                                    edgecolor='none', alpha=0.9)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor('#2171b5' if left >= active_thresh else '#c9a96e')

    ax.axvline(manual_thresh, color='red', lw=1.5, ls='--',
               label=f'Manual threshold = {manual_thresh:.2f}')
    if otsu_thresh is not None:
        ax.axvline(otsu_thresh, color='green', lw=1.5, ls='-.',
                   label=f'Otsu threshold = {otsu_thresh:.3f}')

    water_pct = (valid >= active_thresh).sum() / len(valid) * 100 if len(valid) else 0
    ax.text(0.98, 0.95, f'Water pixels: {water_pct:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            color='#2171b5', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    date_fmt = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    ax.set_title(f'NDWI — {date_fmt}', fontsize=11)
    ax.set_xlabel('NDWI'); ax.set_ylabel('Pixel count')
    ax.set_xlim(-1, 1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def make_folium_map(gdfs_wgs84, aoi_geojson, centroid_lat, centroid_lon):
    fmap = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=15)
    folium.GeoJson(aoi_geojson, name='AOI',
                   style_function=lambda _: {'color':'red','fill':False,'weight':2}
                   ).add_to(fmap)
    colors = ['#08519c','#2171b5','#4292c6','#6baed6','#9ecae1']
    for i, (ds, gdf) in enumerate(gdfs_wgs84.items()):
        if not gdf.empty:
            c = colors[i % len(colors)]
            folium.GeoJson(gdf.__geo_interface__, name=f'Water {ds}',
                style_function=lambda _, col=c: {
                    'fillColor':col,'color':col,'fillOpacity':0.55,'weight':1}
            ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap

def zip_shapefiles(output_dir):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for ext in ('shp','shx','dbf','prj','cpg'):
            for f in output_dir.glob(f'*.{ext}'):
                zf.write(f, f.name)
    return buf.getvalue()

def zip_geotiffs(output_dir):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.glob('NDWI_*.tif'):
            zf.write(f, f.name)
    return buf.getvalue()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['📥 Download', '📊 Process & Analyse', '🗺️ Results'])

# ══════════════════════════════════════════════
# TAB 1 — DOWNLOAD
# ══════════════════════════════════════════════
with tab1:
    st.header('Download PlanetScope Imagery')
    status_box = st.empty()

    def update_status(msg):
        status_box.info(msg)

    if order_mode == 'Use existing order':
        st.info(f'Order ID: `{existing_order_id}`')
        if st.button('Download order', type='primary'):
            with st.spinner('Connecting to Planet...'):
                try:
                    run(_download_existing(api_key, existing_order_id, update_status))
                    st.success('Download complete!')
                    st.session_state.download_done = True
                except Exception as ex:
                    st.error(f'Download failed: {ex}')
    else:
        if st.button('Search for scenes'):
            with st.spinner('Searching Planet catalogue...'):
                try:
                    items = run(_search(api_key, aoi_geojson,
                                        start_date_val, end_date_val, cloud_pct))
                    if items:
                        import pandas as pd
                        df = pd.DataFrame([{
                            'Scene ID': it['id'],
                            'Acquired': it['properties']['acquired'][:10],
                            'Cloud %':  round(it['properties']['cloud_cover']*100, 1),
                        } for it in items])
                        st.dataframe(df, use_container_width=True)
                        st.session_state['_found_items'] = items
                    else:
                        st.warning('No scenes found. Try adjusting date or cloud cover.')
                except Exception as ex:
                    st.error(f'Search failed: {ex}')

        if st.session_state.get('_found_items'):
            if st.button('Order & download selected scenes', type='primary'):
                item_ids = [it['id'] for it in st.session_state['_found_items']]
                with st.spinner('Ordering and waiting for Planet to process...'):
                    try:
                        run(_order_and_download(api_key, item_ids,
                                                aoi_geojson, update_status))
                        st.success('Download complete!')
                        st.session_state.download_done = True
                    except Exception as ex:
                        st.error(f'Order/download failed: {ex}')

    # Discover TIFs regardless of how they got there
    tif_files = sorted(DIR_DOWNLOAD.rglob('*_SR_*.tif'))
    if not tif_files:
        tif_files = sorted(f for f in DIR_DOWNLOAD.rglob('*.tif')
                           if 'NDWI' not in f.stem and 'NDWIMask' not in f.stem)
    st.session_state.tif_files = tif_files

    if tif_files:
        st.success(f'{len(tif_files)} TIF file(s) ready to process.')
        with st.expander('Show files'):
            for f in tif_files:
                st.code(str(f))

# ══════════════════════════════════════════════
# TAB 2 — PROCESS & ANALYSE
# ══════════════════════════════════════════════
with tab2:
    st.header('NDWI Processing & Threshold')

    c1, c2 = st.columns([1, 2])
    with c1:
        thresh_mode = st.radio('Threshold method',
                               ['Manual', 'Otsu auto-threshold'],
                               help='Otsu automatically finds the optimal '
                                    'water/non-water split for each scene.')
        manual_thresh = st.slider('Manual NDWI threshold', -1.0, 1.0, 0.1,
                                  step=0.01,
                                  disabled=(thresh_mode == 'Otsu auto-threshold'))
        if thresh_mode == 'Otsu auto-threshold':
            st.info('Otsu threshold is computed per scene after processing. '
                    'The manual threshold line is still shown for reference.')

    with c2:
        non_null = {ds: tv for ds, tv in st.session_state.otsu_values.items()
                    if tv is not None}
        if thresh_mode == 'Otsu auto-threshold' and non_null:
            st.subheader('Otsu thresholds')
            for ds, tv in non_null.items():
                date_fmt = f'{ds[:4]}-{ds[4:6]}-{ds[6:]}'
                st.metric(date_fmt, f'{tv:.3f}')

    st.divider()

    def _apply_threshold(ndwi_arrays, thresh_mode, manual_thresh, ndwi_dir, shp_dir):
        """Re-apply threshold to cached NDWI arrays — fast, no TIF re-read."""
        otsu_values, gdfs_native = {}, {}
        for date_str, ndwi in ndwi_arrays.items():
            active_thresh = otsu_threshold(ndwi) if thresh_mode == 'Otsu auto-threshold' \
                            else manual_thresh
            otsu_values[date_str] = active_thresh if thresh_mode == 'Otsu auto-threshold' \
                                    else None

            with rasterio.open(ndwi_dir / f'NDWI_{date_str}.tif') as src:
                profile = src.profile.copy()
                transform, crs = src.transform, src.crs

            water = np.where(np.isfinite(ndwi) & (ndwi > active_thresh), 1, 0).astype(np.uint8)
            mask_path = ndwi_dir / f'NDWIMask_{date_str}.tif'
            p = profile.copy(); p.update(count=1, dtype='uint8', nodata=0)
            with rasterio.open(mask_path, 'w', **p) as dst:
                dst.write(water, 1)

            polys = [shape(g) for g, v in shapes(water, transform=transform) if v == 1]
            if polys:
                gdf = gpd.GeoDataFrame({'date': date_str, 'geometry': polys}, crs=crs)
                gdf = gdf[gdf.geometry.area > min_area_m2].reset_index(drop=True)
            else:
                gdf = gpd.GeoDataFrame(columns=['date', 'geometry'], crs=crs)
            gdfs_native[date_str] = gdf

        return otsu_values, gdfs_native

    def _save_results(gdfs_native, otsu_values, ndwi_dir, shp_dir):
        """Write shapefile to disk and build in-memory zips."""
        st.session_state.otsu_values = otsu_values
        st.session_state.gdfs_wgs84  = {d: g.to_crs('EPSG:4326')
                                         for d, g in gdfs_native.items()
                                         if not g.empty}
        combined = gpd.pd.concat(
            [g.to_crs(_src_epsg) for g in gdfs_native.values() if not g.empty],
            ignore_index=True)
        if not combined.empty:
            # Build threshold label for filename
            non_null_thresh = [v for v in otsu_values.values() if v is not None]
            if non_null_thresh:
                shp_name = 'NDWIMask_water_planet_otsu.shp'
            else:
                shp_name = f'NDWIMask_water_planet_{manual_thresh:.3f}.shp'
            combined.to_file(shp_dir / shp_name)
        st.session_state.zip_shp_bytes = zip_shapefiles(shp_dir)
        st.session_state.zip_tif_bytes = zip_geotiffs(ndwi_dir)
        st.session_state.process_done  = True

    if not st.session_state.tif_files:
        st.warning('No TIF files found. Go to the Download tab first.')
    else:
        bc1, bc2 = st.columns(2)

        # ── Step 1: Compute NDWI (slow — reads TIFs) ──────────────────
        if bc1.button('Compute NDWI', type='primary',
                      help='Reads TIF files and computes NDWI. Only needed once per download.'):
            ndwi_arrays = {}
            prog = st.progress(0)
            for i, tif in enumerate(st.session_state.tif_files):
                date_str = tif.stem[:8] if tif.stem[:8].isdigit() else tif.stem
                st.write(f'Reading `{tif.name}`...')
                try:
                    ndwi, _ = compute_ndwi(tif)
                except ValueError as ve:
                    st.warning(f'Skipped {tif.name}: {ve}')
                    prog.progress((i + 1) / len(st.session_state.tif_files))
                    continue
                with rasterio.open(tif) as src:
                    profile = src.profile.copy()
                profile.update(count=1, dtype='float32', nodata=np.nan)
                ndwi_path = DIR_NDWI / f'NDWI_{date_str}.tif'
                with rasterio.open(ndwi_path, 'w', **profile) as dst:
                    dst.write(ndwi.astype('float32'), 1)
                ndwi_arrays[date_str] = ndwi
                prog.progress((i + 1) / len(st.session_state.tif_files))
            st.session_state.ndwi_arrays = ndwi_arrays
            otsu_values, gdfs_native = _apply_threshold(
                ndwi_arrays, thresh_mode, manual_thresh, DIR_NDWI, DIR_SHP)
            _save_results(gdfs_native, otsu_values, DIR_NDWI, DIR_SHP)
            st.success('NDWI computed. Adjust threshold below then click Apply Threshold.')

        # ── Step 2: Re-apply threshold (fast — uses cached arrays) ────
        if st.session_state.ndwi_arrays:
            if bc2.button('Apply Threshold',
                          help='Re-applies current threshold to cached NDWI. Fast — no TIF re-read.'):
                otsu_values, gdfs_native = _apply_threshold(
                    st.session_state.ndwi_arrays, thresh_mode, manual_thresh, DIR_NDWI, DIR_SHP)
                _save_results(gdfs_native, otsu_values, DIR_NDWI, DIR_SHP)
                st.success(f'Threshold applied — shapefile updated → `{DIR_SHP}`')
        else:
            bc2.button('Apply Threshold', disabled=True,
                       help='Compute NDWI first.')

    # Histograms (shown after processing or on rerun if arrays exist)
    if st.session_state.ndwi_arrays:
        st.subheader('NDWI Histograms')
        for date_str, ndwi in st.session_state.ndwi_arrays.items():
            otsu_val = st.session_state.otsu_values.get(date_str)
            with st.expander(f'Scene: {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}',
                             expanded=True):
                fig = build_histogram(date_str, ndwi, manual_thresh, otsu_val)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)

# ══════════════════════════════════════════════
# TAB 3 — RESULTS
# ══════════════════════════════════════════════
with tab3:
    st.header('Results')

    if not st.session_state.process_done:
        st.info('Run processing in the Process & Analyse tab first.')
    else:
        # Map
        fmap = make_folium_map(
            st.session_state.gdfs_wgs84,
            st.session_state.aoi_geojson,
            st.session_state.centroid_lat,
            st.session_state.centroid_lon,
        )
        st_folium(fmap, width=None, height=520, returned_objects=[])

        st.divider()

        # Summary table
        rows = []
        for ds, gdf in st.session_state.gdfs_wgs84.items():
            area = gdf.to_crs('EPSG:7850').geometry.area.sum() if not gdf.empty else 0
            rows.append({
                'Date': f'{ds[:4]}-{ds[4:6]}-{ds[6:]}',
                'Water polygons': len(gdf),
                'Total water area (m²)': f'{area:,.0f}',
                'Otsu threshold': f'{st.session_state.otsu_values[ds]:.3f}'
                                  if st.session_state.otsu_values.get(ds) is not None else '—',
            })
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()

        # Download buttons
        dc1, dc2 = st.columns(2)
        if st.session_state.zip_shp_bytes:
            dc1.download_button(
                '⬇️ Download Shapefile (ZIP)',
                data=st.session_state.zip_shp_bytes,
                file_name='water_mask_planet.zip',
                mime='application/zip',
            )
        if st.session_state.zip_tif_bytes:
            dc2.download_button(
                '⬇️ Download NDWI GeoTIFFs (ZIP)',
                data=st.session_state.zip_tif_bytes,
                file_name='ndwi_rasters_planet.zip',
                mime='application/zip',
            )
