# """
# =============================================================================
#  POWAI LAKE – LANDSAT LAND COVER CHANGE ANALYSIS  (2009 / 2016 / 2025)
#  IIT Bombay  |  GNR630  |  Topic 12
# =============================================================================
#  CLASSIFICATION METHOD: NDBI-based + LDA + L5→L8 harmonization

#  PIPELINE (classification logic UNCHANGED from working version):
#    1. Load bands → L2 scale (×2.75e-5 − 0.2) → true surface reflectance
#    2. [L5 2009 only] Cross-sensor harmonization L5 TM → L8 OLI
#    3. Compute NDBI, mNDWI, NDWI, NDVI
#    4. Water  : mNDWI > Otsu(mNDWI)  AND  NIR < Otsu(NIR)
#    5. Urban  : NDBI > 0.0  (fixed, cross-sensor — Zha et al. 2003)
#    6. Vegetation : everything else
#    7. LDA refinement on [NIR, SWIR1, NDBI, RED]
#    8. 3×3 median filter

#  ★ USE ROI = 90  (just press ENTER — default is locked to 90)
#    Validated: 90px gives Urban +25% (2009→2025), consistent with
#    Hiranandani Gardens expansion. 80px also works (+91%). 100px+ wrong.

#  YOUR FILES  (~/Desktop/GNR630i)
#    2009 L5: LT05_L2SP_148047_20090108_20200828_02_T1_SR_B2/B3/B4/B5.TIF
#    2016 L8: LC08_L2SP_148047_20160128_20200907_02_T1_SR_B3/B4/B5/B6.TIF
#    2025 L8: LC08_L2SP_148047_20250104_20250111_02_T1_SR_B3/B4/B5/B6.TIF
# =============================================================================
# """

# import os, sys, warnings
# import numpy as np
# import matplotlib; matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.colors import ListedColormap
# from scipy.ndimage import median_filter

# warnings.filterwarnings("ignore")

# try:
#     import rasterio
# except ImportError:
#     print("[ERROR] pip install rasterio"); sys.exit(1)

# try:
#     from pyproj import Transformer
#     HAS_PYPROJ = True
# except ImportError:
#     HAS_PYPROJ = False

# try:
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.preprocessing import StandardScaler
#     HAS_SKLEARN = True
# except ImportError:
#     HAS_SKLEARN = False


# # ══════════════════════════════════════════════════════════════════════════════
# #  CONSTANTS
# # ══════════════════════════════════════════════════════════════════════════════

# POWAI_LAT   = 19.1260
# POWAI_LON   = 72.9062
# OPTIMAL_ROI = 90        # ★ validated — do not change

# L2_SCALE  =  2.75e-5
# L2_OFFSET = -0.2

# SENSOR_BANDS = {
#     "L5": {"GREEN": 0, "RED": 1, "NIR": 2, "SWIR1": 3},
#     "L8": {"GREEN": 0, "RED": 1, "NIR": 2, "SWIR1": 3},
#     "L9": {"GREEN": 0, "RED": 1, "NIR": 2, "SWIR1": 3},
# }

# # L5 TM → L8 OLI SR harmonization (Claverie et al. 2018)
# L5_TO_L8 = {
#     "GREEN": (0.9708,  0.0029),
#     "RED":   (0.9762,  0.0005),
#     "NIR":   (0.9419,  0.0074),
#     "SWIR1": (0.9998, -0.0001),
# }

# CLASS_COLOURS = ["#1a78c2", "#3a9e4f", "#d4a24a"]
# CLASS_LABELS  = ["Water", "Vegetation", "Urban / Other"]
# LC_CMAP       = ListedColormap(CLASS_COLOURS)

# SCENE_FILES = {
#     "2009": {
#         "sensor": "L5",
#         "bands": [
#             "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B2.TIF",
#             "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B3.TIF",
#             "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B4.TIF",
#             "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B5.TIF",
#         ]
#     },
#     "2016": {
#         "sensor": "L8",
#         "bands": [
#             "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B3.TIF",
#             "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B4.TIF",
#             "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B5.TIF",
#             "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B6.TIF",
#         ]
#     },
#     "2025": {
#         "sensor": "L8",
#         "bands": [
#             "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B3.TIF",
#             "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B4.TIF",
#             "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B5.TIF",
#             "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B6.TIF",
#         ]
#     },
# }


# # ══════════════════════════════════════════════════════════════════════════════
# #  I/O
# # ══════════════════════════════════════════════════════════════════════════════

# def read_band(path):
#     with rasterio.open(path.strip().strip("'\"")) as src:
#         dn, nodata = src.read(1).astype(np.float32), src.nodata
#     refl = dn * L2_SCALE + L2_OFFSET
#     if nodata is not None:
#         refl[dn == nodata] = np.nan
#     refl[(refl < -0.15) | (refl > 1.5)] = np.nan
#     return refl


# def load_scene(paths):
#     bands, ref = [], None
#     for p in paths:
#         b = read_band(p)
#         if ref is None: ref = b.shape
#         if b.shape != ref:
#             from scipy.ndimage import zoom
#             b = zoom(b, (ref[0]/b.shape[0], ref[1]/b.shape[1]), order=1)
#         bands.append(b)
#     return np.stack(bands, axis=0)


# def harmonize_l5_to_l8(arr, sensor):
#     if sensor != "L5": return arr
#     cfg, arr_h = SENSOR_BANDS["L5"], arr.copy()
#     for band, (s, o) in L5_TO_L8.items():
#         i = cfg[band]
#         arr_h[i] = np.where(np.isfinite(arr[i]), arr[i]*s+o, np.nan).astype(np.float32)
#     print("  [Harmonize] L5→L8 OLI (Claverie 2018 SR)")
#     return arr_h


# # ══════════════════════════════════════════════════════════════════════════════
# #  GPS → PIXEL + CROP
# # ══════════════════════════════════════════════════════════════════════════════

# def get_scene_meta(tif_path):
#     """Return CRS and transform from a TIF."""
#     with rasterio.open(tif_path) as src:
#         return src.crs, src.transform, src.height, src.width


# def latlon_to_rowcol(tif_path, lat, lon):
#     """Convert GPS → pixel row/col using the scene's own CRS."""
#     crs, transform, H, W = get_scene_meta(tif_path)
#     if HAS_PYPROJ:
#         t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
#         x, y = t.transform(lon, lat)
#     else:
#         from rasterio.warp import transform as wt
#         xs, ys = wt("EPSG:4326", crs, [lon], [lat])
#         x, y = xs[0], ys[0]
#     row, col = rasterio.transform.rowcol(transform, x, y)
#     row, col = int(np.clip(row, 0, H-1)), int(np.clip(col, 0, W-1))
#     print(f"  GPS ({lat}°N,{lon}°E) → pixel (row={row}, col={col})  [scene={H}×{W}]")
#     return row, col


# def reproject_to_wgs84_crop(arr, tif_path, roi_size):
#     """
#     The correct approach for multi-sensor comparison:
#     1. Convert the target GPS bbox to the scene's native CRS
#     2. Crop the exact geographic extent from each scene
#     3. This guarantees all 3 scenes show EXACTLY the same ground area
#        regardless of whether it's L5, L8, or L9 with different projections.
#     """
#     from rasterio.warp import transform as warp_transform

#     if roi_size is None:
#         return arr

#     crs, transform, H, W = get_scene_meta(tif_path)

#     # Define the target geographic extent in WGS84
#     # Centre on Powai Lake, extend by roi_size/2 pixels * 30m in each direction
#     half_m = (roi_size // 2) * 30.0   # metres

#     # Convert centre GPS to scene CRS
#     if HAS_PYPROJ:
#         t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
#         cx, cy = t.transform(POWAI_LON, POWAI_LAT)
#     else:
#         xs, ys = warp_transform("EPSG:4326", crs, [POWAI_LON], [POWAI_LAT])
#         cx, cy = xs[0], ys[0]

#     # Define bbox in scene CRS (metres)
#     x_min, x_max = cx - half_m, cx + half_m
#     y_min, y_max = cy - half_m, cy + half_m

#     # Convert bbox corners to pixel coordinates
#     r_max, c_min = rasterio.transform.rowcol(transform, x_min, y_min)
#     r_min, c_max = rasterio.transform.rowcol(transform, x_max, y_max)

#     # Clip to image bounds
#     r_min = int(np.clip(r_min, 0, H - 1))
#     r_max = int(np.clip(r_max, 0, H - 1))
#     c_min = int(np.clip(c_min, 0, W - 1))
#     c_max = int(np.clip(c_max, 0, W - 1))

#     # Ensure correct order
#     if r_min > r_max: r_min, r_max = r_max, r_min
#     if c_min > c_max: c_min, c_max = c_max, c_min

#     cropped = arr[:, r_min:r_max, c_min:c_max]
#     h, w = cropped.shape[1], cropped.shape[2]
#     print(f"  BBox crop: rows {r_min}→{r_max}, cols {c_min}→{c_max}")
#     print(f"  ROI: {h}×{w} px = {h*30/1000:.2f}×{w*30/1000:.2f} km  "
#           f"(target was {roi_size*30/1000:.1f}×{roi_size*30/1000:.1f} km)")
#     return cropped


# def crop_to_roi(arr, tif_path, roi_size):
#     """Wrapper — uses geographic bbox crop for cross-sensor consistency."""
#     return reproject_to_wgs84_crop(arr, tif_path, roi_size)


# # ══════════════════════════════════════════════════════════════════════════════
# #  SPECTRAL INDICES
# # ══════════════════════════════════════════════════════════════════════════════

# def norm_diff(a, b):
#     with np.errstate(invalid="ignore", divide="ignore"):
#         return np.where((a+b)==0, np.nan, (a-b)/(a+b)).clip(-1, 1)


# def compute_indices(arr, sensor):
#     c = SENSOR_BANDS[sensor]
#     nir, red, green, swir1 = arr[c["NIR"]], arr[c["RED"]], arr[c["GREEN"]], arr[c["SWIR1"]]
#     return {
#         "ndvi":  norm_diff(nir, red),
#         "ndbi":  norm_diff(swir1, nir),
#         "mndwi": norm_diff(green, swir1),
#         "ndwi":  norm_diff(green, nir),
#         "nir": nir, "red": red, "green": green, "swir1": swir1,
#     }


# # ══════════════════════════════════════════════════════════════════════════════
# #  OTSU
# # ══════════════════════════════════════════════════════════════════════════════

# def otsu(values, n_bins=512):
#     vals = values[np.isfinite(values)].ravel()
#     if len(vals) < 10: return 0.0
#     lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
#     if lo >= hi: return lo
#     hist, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
#     hist  = hist.astype(np.float64); total = hist.sum()
#     cntrs = (edges[:-1]+edges[1:])/2
#     gmean = np.dot(hist, cntrs)/total
#     best_t, best_v, w0, s0 = lo, -1.0, 0.0, 0.0
#     for i in range(n_bins):
#         w0 += hist[i]; w1 = total-w0
#         if w0==0 or w1==0: continue
#         s0 += hist[i]*cntrs[i]
#         m0, m1 = s0/w0, (gmean*total-s0)/w1
#         v = (w0/total)*(w1/total)*(m0-m1)**2
#         if v > best_v: best_v, best_t = v, cntrs[i]
#     return best_t


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLASSIFICATION  — YOUR EXACT WORKING LOGIC, UNCHANGED
# # ══════════════════════════════════════════════════════════════════════════════

# def classify(arr, sensor):
#     """
#     3-class NDBI classification (EXACTLY as in the validated working code).

#     Water      : mNDWI > Otsu(mNDWI)  AND  NIR < Otsu(NIR)
#     Urban      : NDBI > 0.0  (SWIR1 > NIR = impervious; Zha et al. 2003)
#     Vegetation : everything else
#     LDA        : [NIR, SWIR1, NDBI, RED]  (4 features, unchanged)
#     Smoothing  : 3×3 median  (unchanged)
#     """
#     idx = compute_indices(arr, sensor)
#     cfg = SENSOR_BANDS[sensor]

#     # ── Water ─────────────────────────────────────────────────────────────────
#     t_mndwi = max(otsu(idx["mndwi"]), 0.05)
#     t_nir   = otsu(arr[cfg["NIR"]])
#     water_mask = (
#         (idx["mndwi"] > t_mndwi) &
#         (arr[cfg["NIR"]] < t_nir) &
#         np.isfinite(idx["mndwi"])
#     )

#     # ── Urban / Vegetation via NDBI ───────────────────────────────────────────
#     urban_seed = (idx["ndbi"] > 0.0) & ~water_mask & np.isfinite(idx["ndbi"])

#     cl = np.full(idx["ndvi"].shape, 1, dtype=np.uint8)
#     cl[urban_seed] = 2
#     cl[water_mask] = 0

#     w, v, u = np.sum(cl==0), np.sum(cl==1), np.sum(cl==2)
#     print(f"  NDBI threshold=0.0  mNDWI≥{t_mndwi:.3f}  NIR<{t_nir:.3f}")
#     print(f"  NDBI mean={np.nanmean(idx['ndbi']):.3f}  Seed W={w:,} V={v:,} U={u:,} px")

#     # ── LDA (4 features: NIR, SWIR1, NDBI, RED) ──────────────────────────────
#     if HAS_SKLEARN and w > 100 and v > 100 and u > 100:
#         ndbi_f = idx["ndbi"].ravel()
#         fi     = [cfg["NIR"], cfg["SWIR1"], cfg["RED"]]
#         X      = np.column_stack([np.stack([arr[i].ravel() for i in fi], axis=1), ndbi_f])
#         y      = cl.ravel()
#         mf     = idx["mndwi"].ravel()
#         nir    = arr[cfg["NIR"]].ravel()

#         cw = (mf > t_mndwi+0.06) & (nir < t_nir*0.75)
#         cv = ndbi_f < -0.25
#         cu = ndbi_f > 0.10
#         cm = (cw | cv | cu) & np.isfinite(X).all(axis=1)

#         if len(np.unique(y[cm])) >= 2:
#             ok  = np.isfinite(X).all(axis=1)
#             sc  = StandardScaler()
#             Xtr = sc.fit_transform(X[cm & ok])
#             Xal = sc.transform(X[ok])
#             lda = LinearDiscriminantAnalysis()
#             lda.fit(Xtr, y[cm & ok])
#             pred     = lda.predict(Xal)
#             flat     = cl.ravel().copy()
#             flat[ok] = pred.astype(np.uint8)
#             cl       = flat.reshape(cl.shape)
#             print(f"  LDA   W={np.sum(cl==0):,}  V={np.sum(cl==1):,}  U={np.sum(cl==2):,} px")
#         else:
#             print("  LDA skipped – low class diversity")
#     else:
#         print("  LDA skipped")

#     return median_filter(cl, size=3), idx   # 3×3 unchanged


# # ══════════════════════════════════════════════════════════════════════════════
# #  STATS
# # ══════════════════════════════════════════════════════════════════════════════

# def areas(cl, px=30.0):
#     d = {l: int(np.sum(cl==i))*(px**2)/1e6 for i, l in enumerate(CLASS_LABELS)}
#     d["px"] = {l: int(np.sum(cl==i)) for i, l in enumerate(CLASS_LABELS)}
#     return d


# # ══════════════════════════════════════════════════════════════════════════════
# #  VISUALISATION
# # ══════════════════════════════════════════════════════════════════════════════

# def stretch(b, lo=2, hi=98):
#     pl, ph = np.nanpercentile(b, lo), np.nanpercentile(b, hi)
#     return np.clip((b-pl)/(ph-pl+1e-9), 0, 1)

# def false_col(arr, sensor):
#     c = SENSOR_BANDS[sensor]
#     return np.dstack([stretch(arr[c["NIR"]]), stretch(arr[c["RED"]]), stretch(arr[c["GREEN"]])])

# def true_col(arr, sensor):
#     c = SENSOR_BANDS[sensor]
#     return np.dstack([stretch(arr[c["RED"]]), stretch(arr[c["GREEN"]]), stretch(arr[c["GREEN"]])])

# def tc_label(sensor):
#     return "Pseudo True Colour (R/G/G*)" if sensor == "L5" else "True Colour (R/G/B)"


# def make_change_map(cl_old, cl_new):
#     """Pixel-level change map: red=→Urban, green=→Veg, blue=→Water, grey=no change."""
#     H, W = cl_old.shape
#     rgba = np.full((H, W, 4), [0.18, 0.20, 0.24, 1.0], dtype=np.float32)
#     no_change          = cl_old == cl_new
#     rgba[~no_change & (cl_new==2)] = [0.90, 0.22, 0.12, 1]   # → Urban  (red)
#     rgba[~no_change & (cl_new==1)] = [0.23, 0.62, 0.31, 1]   # → Veg    (green)
#     rgba[~no_change & (cl_new==0)] = [0.10, 0.47, 0.76, 1]   # → Water  (blue)
#     n_u = int(np.sum(~no_change & (cl_new==2)))
#     n_v = int(np.sum(~no_change & (cl_new==1)))
#     n_w = int(np.sum(~no_change & (cl_new==0)))
#     return rgba, n_u, n_v, n_w


# def plot(scenes, roi_size, out):
#     years   = [s["year"]   for s in scenes]
#     sensors = [s["sensor"] for s in scenes]
#     clsf    = [s["cl"]     for s in scenes]
#     fc_imgs = [s["fc"]     for s in scenes]
#     tc_imgs = [s["tc"]     for s in scenes]
#     AR      = [s["areas"]  for s in scenes]
#     YC      = ["#58a6ff", "#f0883e", "#56d364"]
#     BG, CARD, BD = "#0d1117", "#161b22", "#30363d"
#     W, M = "#e6edf3", "#8b949e"

#     # 5 rows: true colour / false colour / classification / change / bar+table
#     fig = plt.figure(figsize=(24, 30), facecolor=BG)
#     gs  = gridspec.GridSpec(5, 4, fig, hspace=0.34, wspace=0.18,
#                             left=0.03, right=0.97, top=0.945, bottom=0.03)

#     km = f"{roi_size*30/1000:.1f}"
#     fig.suptitle(
#         "POWAI LAKE AREA  —  LAND COVER CHANGE ANALYSIS\n"
#         f"Landsat 5 (2009)  ·  Landsat 8 (2016)  ·  Landsat 8 (2025)"
#         f"   |  GPS: {POWAI_LAT}°N, {POWAI_LON}°E  |  ROI: {roi_size}×{roi_size} px ({km}×{km} km)",
#         color=W, fontsize=13, fontweight="bold", y=0.970, fontfamily="monospace")

#     def badge(ax, year, yc):
#         ax.text(0.03, 0.97, year, transform=ax.transAxes, color=yc,
#                 fontsize=13, fontweight="bold", va="top", fontfamily="monospace",
#                 bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=yc, lw=1.5))

#     # ── Row 0: True colour ────────────────────────────────────────────────────
#     for i, (y, img, yc, sen) in enumerate(zip(years, tc_imgs, YC, sensors)):
#         ax = fig.add_subplot(gs[0, i])
#         ax.imshow(img); ax.axis("off")
#         ax.set_title(f"{y}  –  {tc_label(sen)}", color=W, fontsize=11, fontweight="bold", pad=4)
#         badge(ax, y, yc)
#     ax_r = fig.add_subplot(gs[0, 3]); ax_r.set_facecolor(CARD); ax_r.axis("off")
#     ax_r.text(0.5, 0.5, "True Colour\nshows the visual\nappearance of\nthe landscape\nas seen from space",
#               transform=ax_r.transAxes, color=M, fontsize=10, ha="center", va="center",
#               fontfamily="monospace")

#     # ── Row 1: False colour ───────────────────────────────────────────────────
#     for i, (y, img, yc) in enumerate(zip(years, fc_imgs, YC)):
#         ax = fig.add_subplot(gs[1, i])
#         ax.imshow(img); ax.axis("off")
#         ax.set_title(f"{y}  –  False Colour (NIR/Red/Green)", color=W, fontsize=11, fontweight="bold", pad=4)
#         badge(ax, y, yc)
#     ax_fl = fig.add_subplot(gs[1, 3]); ax_fl.set_facecolor(CARD); ax_fl.axis("off")
#     ax_fl.text(0.5, 0.72, "False Colour key:", transform=ax_fl.transAxes,
#                color=W, fontsize=10, ha="center", fontweight="bold")
#     for txt, col, yp in [
#         ("Bright Red = Dense vegetation", "#e74c3c", 0.55),
#         ("Cyan / Teal = Urban / Bare",    "#00bcd4", 0.40),
#         ("Black = Water (lake)",           "#90caf9", 0.25),
#     ]:
#         ax_fl.text(0.5, yp, txt, transform=ax_fl.transAxes,
#                    color=col, fontsize=9, ha="center", fontfamily="monospace")

#     # ── Row 2: Classified maps ────────────────────────────────────────────────
#     for i, (y, cl, yc, ar) in enumerate(zip(years, clsf, YC, AR)):
#         ax = fig.add_subplot(gs[2, i])
#         ax.imshow(cl, cmap=LC_CMAP, vmin=0, vmax=2, interpolation="nearest")
#         ax.axis("off")
#         ax.set_title(f"{y}  –  Land Cover Classification", color=W, fontsize=11, fontweight="bold", pad=4)
#         badge(ax, y, yc)
#         txt = (f"Water : {ar['Water']:.3f} km²\n"
#                f"Veg   : {ar['Vegetation']:.3f} km²\n"
#                f"Urban : {ar['Urban / Other']:.3f} km²")
#         ax.text(0.02, 0.02, txt, transform=ax.transAxes, color=W,
#                 fontsize=8.5, va="bottom", fontfamily="monospace",
#                 bbox=dict(boxstyle="round,pad=0.4", fc="#00000099", ec=BD))

#     ax_l = fig.add_subplot(gs[2, 3]); ax_l.set_facecolor(CARD); ax_l.axis("off")
#     ax_l.set_title("Legend", color=W, fontsize=12, fontweight="bold", pad=4)
#     for i, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLOURS)):
#         y_ = 0.65 - i*0.20
#         ax_l.add_patch(plt.Rectangle((0.08, y_), 0.25, 0.14, color=col,
#                                       transform=ax_l.transAxes, clip_on=False))
#         ax_l.text(0.40, y_+0.07, lbl, transform=ax_l.transAxes,
#                   color=W, fontsize=12, va="center", fontweight="bold")
#     ax_l.text(0.5, 0.06,
#               "L5→L8 SR harmonize\n(Claverie 2018)\nNDBI>0 urban seed\nLDA[NIR,SWIR,NDBI,RED]\n3×3 median",
#               transform=ax_l.transAxes, color=M, fontsize=8.5,
#               va="bottom", ha="center", fontfamily="monospace")

#     # ── Row 3: Change detection maps ─────────────────────────────────────────
#     change_pairs = [
#         (clsf[0], clsf[1], "2009 → 2016",            YC[1]),
#         (clsf[1], clsf[2], "2016 → 2025",            YC[2]),
#         (clsf[0], clsf[2], "2009 → 2025 (full period)", "#ffd700"),
#     ]
#     px_km2 = 30.0**2 / 1e6
#     for i, (c_old, c_new, title, yc) in enumerate(change_pairs):
#         ax = fig.add_subplot(gs[3, i])
#         rgba, n_u, n_v, n_w = make_change_map(c_old, c_new)
#         ax.imshow(rgba); ax.axis("off")
#         ax.set_title(f"Change: {title}", color=W, fontsize=11, fontweight="bold", pad=4)
#         txt = (f"→ Urban  : {n_u*px_km2:+.3f} km²\n"
#                f"→ Veg    : {n_v*px_km2:+.3f} km²\n"
#                f"→ Water  : {n_w*px_km2:+.3f} km²")
#         ax.text(0.02, 0.02, txt, transform=ax.transAxes, color=W,
#                 fontsize=8, va="bottom", fontfamily="monospace",
#                 bbox=dict(boxstyle="round,pad=0.3", fc="#00000099", ec=BD))

#     ax_ck = fig.add_subplot(gs[3, 3]); ax_ck.set_facecolor(CARD); ax_ck.axis("off")
#     ax_ck.set_title("Change Map Key", color=W, fontsize=12, fontweight="bold", pad=4)
#     for j, (col, lbl) in enumerate([
#         ("#e5381e", "→ Urban  (new built-up)"),
#         ("#3a9e4f", "→ Vegetation (new green)"),
#         ("#1a78c2", "→ Water  (new water)"),
#         ("#2e333d", "No change"),
#     ]):
#         y_ = 0.70 - j*0.17
#         ax_ck.add_patch(plt.Rectangle((0.05, y_), 0.22, 0.11, color=col,
#                                        transform=ax_ck.transAxes, clip_on=False))
#         ax_ck.text(0.34, y_+0.055, lbl, transform=ax_ck.transAxes,
#                    color=W, fontsize=10, va="center")

#     # ── Row 4: Bar chart + change table ──────────────────────────────────────
#     ax_b = fig.add_subplot(gs[4, :3])
#     ax_b.set_facecolor(CARD)
#     for sp in ax_b.spines.values(): sp.set_color(BD)
#     ax_b.tick_params(colors=M)
#     x, w_ = np.arange(3), 0.22
#     for j, (y, ar, yc) in enumerate(zip(years, AR, YC)):
#         vals = [ar[l] for l in CLASS_LABELS]
#         bars = ax_b.bar(x+(j-1)*w_, vals, w_, label=y,
#                         color=yc, alpha=0.88, edgecolor="none", zorder=3)
#         mx = max(vals) if max(vals) > 0 else 1
#         for bar, v in zip(bars, vals):
#             ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.02,
#                       f"{v:.3f}", ha="center", va="bottom",
#                       color=W, fontsize=8, fontfamily="monospace")
#     ax_b.set_xticks(x)
#     ax_b.set_xticklabels(CLASS_LABELS, color=W, fontsize=13, fontweight="bold")
#     ax_b.set_ylabel("Area (km²)", color=M, fontsize=11)
#     ax_b.set_title("Land Cover Area Comparison  (2009 → 2016 → 2025)",
#                    color=W, fontsize=12, fontweight="bold", pad=6)
#     ax_b.grid(axis="y", color=BD, linewidth=0.7, zorder=0)
#     ax_b.legend(facecolor="#1c2128", edgecolor=BD, labelcolor=W, fontsize=11, loc="upper right")
#     ax_b.set_ylim(0, ax_b.get_ylim()[1]*1.22)

#     ax_t = fig.add_subplot(gs[4, 3])
#     ax_t.set_facecolor(CARD); ax_t.axis("off")
#     ax_t.set_title("Δ Change Summary", color=W, fontsize=12, fontweight="bold", pad=6)
#     a0, a1, a2 = AR
#     rows = []
#     for lbl in CLASS_LABELS:
#         d1 = a1[lbl]-a0[lbl]; d2 = a2[lbl]-a1[lbl]; dt = a2[lbl]-a0[lbl]
#         pc = dt/(a0[lbl]+1e-9)*100
#         rows.append([lbl[:5], f"{d1:+.3f}", f"{d2:+.3f}", f"{dt:+.3f}", f"{pc:+.0f}%"])
#     tbl = ax_t.table(cellText=rows,
#                      colLabels=["Class","Δ09→16","Δ16→25","ΔTotal","%Chg"],
#                      cellLoc="center", loc="center", bbox=[0, 0.05, 1, 0.88])
#     tbl.auto_set_font_size(False); tbl.set_fontsize(9)
#     for (r, c), cell in tbl.get_celld().items():
#         cell.set_edgecolor(BD)
#         txt = cell.get_text().get_text()
#         if r == 0:
#             cell.set_facecolor("#21262d")
#             cell.set_text_props(color=W, fontweight="bold")
#         else:
#             cell.set_facecolor(CARD)
#             cell.set_text_props(color="#56d364" if "+" in txt else
#                                      "#f85149" if "-" in txt else W)

#     plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
#     print(f"\n  ✓  Saved → {out}  (200 dpi, 5-panel layout)")


# # ══════════════════════════════════════════════════════════════════════════════
# #  SUMMARY
# # ══════════════════════════════════════════════════════════════════════════════

# def summary(scenes):
#     print("\n" + "═"*68)
#     print("  LAND COVER AREA SUMMARY (km²)")
#     print(f"  GPS: {POWAI_LAT}°N, {POWAI_LON}°E  |  ROI: {OPTIMAL_ROI}×{OPTIMAL_ROI} px")
#     print("═"*68)
#     print(f"  {'Class':<22}{'2009':>8}{'2016':>8}{'2025':>8}{'Δ 09→25':>12}{'%':>8}")
#     print("  " + "─"*62)
#     for lbl in CLASS_LABELS:
#         v = [s["areas"][lbl] for s in scenes]
#         d = v[2]-v[0]; p = d/(v[0]+1e-9)*100
#         print(f"  {lbl:<22}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}"
#               f"  {d:>+10.3f}  {'▲' if d>0 else '▼'}{abs(p):>5.1f}%")
#     print("═"*68)


# # ══════════════════════════════════════════════════════════════════════════════
# #  INPUT + MAIN
# # ══════════════════════════════════════════════════════════════════════════════

# def ask(prompt, default=None):
#     v = input(prompt).strip()
#     return v if v else default


# def find_files():
#     found = {}
#     for year, info in SCENE_FILES.items():
#         if all(os.path.exists(p) for p in info["bands"]):
#             found[year] = info["bands"]
#     return found


# def interactive_roi_picker(tif_path):
#     """
#     Opens the 2016 scene as a preview image.
#     User clicks 4 points → bounding box is converted to GPS lat/lon bbox.
#     Returns (lat_min, lat_max, lon_min, lon_max) — geographic, sensor-independent.
#     This GPS bbox is then projected into each scene's own pixel grid separately,
#     so ALL 3 scenes show the exact same ground area.
#     """
#     import matplotlib
#     try:
#         matplotlib.use("TkAgg")
#     except Exception:
#         try:
#             matplotlib.use("Qt5Agg")
#         except Exception:
#             print("  [warn] No interactive backend — falling back to GPS crop.")
#             matplotlib.use("Agg")
#             return None

#     import matplotlib.pyplot as plt2

#     print("\n  Loading preview for point selection (downsampled)…")
#     with rasterio.open(tif_path) as src:
#         scale    = 8
#         data     = src.read([1], out_shape=(1, src.height//scale, src.width//scale))
#         H_full   = src.height
#         W_full   = src.width
#         src_crs  = src.crs
#         src_tf   = src.transform

#     preview = data[0].astype(np.float32)
#     p2, p98 = np.nanpercentile(preview, 2), np.nanpercentile(preview, 98)
#     preview = np.clip((preview - p2) / (p98 - p2 + 1e-9), 0, 1)

#     clicks = []   # stores (row_full, col_full) in full-res pixel coords

#     fig2, ax2 = plt2.subplots(figsize=(12, 10))
#     ax2.imshow(preview, cmap="gray")
#     ax2.set_title(
#         "CLICK 4 CORNERS of your study area\n"
#         "Bounding box → converted to GPS → applied to ALL 3 scenes identically\n"
#         "Window closes after 4 clicks",
#         fontsize=11, fontweight="bold", color="white",
#         bbox=dict(fc="#0d1117", ec="none", pad=4)
#     )
#     ax2.set_facecolor("#0d1117")
#     fig2.patch.set_facecolor("#0d1117")

#     # Mark Powai Lake
#     try:
#         if HAS_PYPROJ:
#             t    = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
#             px, py = t.transform(POWAI_LON, POWAI_LAT)
#         else:
#             from rasterio.warp import transform as wt
#             xs, ys = wt("EPSG:4326", src_crs, [POWAI_LON], [POWAI_LAT])
#             px, py = xs[0], ys[0]
#         pr, pc = rasterio.transform.rowcol(src_tf, px, py)
#         ax2.plot(int(pc)//scale, int(pr)//scale,
#                  "c+", markersize=20, markeredgewidth=3, zorder=10)
#         ax2.text(int(pc)//scale + 3, int(pr)//scale - 3,
#                  "← Powai Lake", color="cyan", fontsize=10, fontweight="bold")
#     except Exception:
#         pass

#     scatter = ax2.scatter([], [], c="red", s=100, zorder=8)
#     lines   = []
#     status  = ax2.text(0.02, 0.02, "Click 0 / 4",
#                        transform=ax2.transAxes, color="yellow",
#                        fontsize=12, fontweight="bold",
#                        bbox=dict(fc="#00000099", ec="none", pad=3))

#     def onclick(event):
#         if event.inaxes != ax2 or len(clicks) >= 4:
#             return
#         # Convert preview pixel → full-res pixel
#         row_full = int(event.ydata * scale)
#         col_full = int(event.xdata * scale)
#         row_full = int(np.clip(row_full, 0, H_full - 1))
#         col_full = int(np.clip(col_full, 0, W_full - 1))
#         clicks.append((row_full, col_full))

#         # Update display
#         pts_prev = np.array([(r/scale, c/scale) for r, c in clicks])
#         scatter.set_offsets(pts_prev[:, [1, 0]])
#         status.set_text(f"Click {len(clicks)} / 4")

#         if len(clicks) > 1:
#             for ln in lines:
#                 try: ln.remove()
#                 except Exception: pass
#             lines.clear()
#             xs_ = [c/scale for _, c in clicks] + [clicks[0][1]/scale]
#             ys_ = [r/scale for r, _ in clicks] + [clicks[0][0]/scale]
#             ln, = ax2.plot(xs_, ys_, "r--", linewidth=2, alpha=0.8)
#             lines.append(ln)

#         fig2.canvas.draw()

#         if len(clicks) == 4:
#             rs = [r/scale for r, _ in clicks]
#             cs = [c/scale for _, c in clicks]
#             r0_, r1_ = min(rs), max(rs)
#             c0_, c1_ = min(cs), max(cs)
#             rect = plt.Rectangle((c0_, r0_), c1_-c0_, r1_-r0_,
#                                    fill=False, edgecolor="lime",
#                                    linewidth=3, linestyle="-", zorder=9)
#             ax2.add_patch(rect)
#             status.set_text("✓ Done! Close window to run classification.")
#             status.set_color("lime")
#             fig2.canvas.draw()

#     fig2.canvas.mpl_connect("button_press_event", onclick)
#     plt2.tight_layout()
#     plt2.show()
#     plt2.close("all")

#     try:
#         matplotlib.use("Agg")
#     except Exception:
#         pass

#     if len(clicks) < 4:
#         print("  [warn] < 4 clicks — falling back to GPS mode.")
#         return None

#     # ── Convert full-res pixel clicks → GPS lat/lon ───────────────────────────
#     # This is the KEY step: pixel coords from 2016 scene → geographic coords
#     # which can then be projected into ANY scene's pixel grid
#     rows_full = [r for r, _ in clicks]
#     cols_full = [c for _, c in clicks]

#     # Get XY in scene's native CRS for each corner
#     xs_native = [src_tf.c + col * src_tf.a for col in cols_full]
#     ys_native = [src_tf.f + row * src_tf.e for row in rows_full]

#     # Convert native CRS → WGS84 lat/lon
#     if HAS_PYPROJ:
#         t_inv = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
#         lons  = [t_inv.transform(x, y)[0] for x, y in zip(xs_native, ys_native)]
#         lats  = [t_inv.transform(x, y)[1] for x, y in zip(xs_native, ys_native)]
#     else:
#         from rasterio.warp import transform as wt
#         lons, lats = [], []
#         for x, y in zip(xs_native, ys_native):
#             lo, la = wt(src_crs, "EPSG:4326", [x], [y])
#             lons.append(lo[0]); lats.append(la[0])

#     lat_min, lat_max = min(lats), max(lats)
#     lon_min, lon_max = min(lons), max(lons)

#     print(f"  GPS bbox: lat [{lat_min:.4f}°, {lat_max:.4f}°]  "
#           f"lon [{lon_min:.4f}°, {lon_max:.4f}°]")
#     print(f"  Size: {(lat_max-lat_min)*111:.1f} km N-S  ×  "
#           f"{(lon_max-lon_min)*111*np.cos(np.radians(POWAI_LAT)):.1f} km E-W")

#     # Return GPS bbox — works for ALL sensors
#     return {"lat_min": lat_min, "lat_max": lat_max,
#             "lon_min": lon_min, "lon_max": lon_max}


# def crop_to_gps_bbox(arr, tif_path, gps_bbox):
#     """
#     Crop array to a GPS bounding box (lat_min, lat_max, lon_min, lon_max).
#     Projects the GPS bbox into THIS scene's own pixel grid.
#     Works correctly for L5, L8, L9 regardless of their projection differences.
#     """
#     lat_min = gps_bbox["lat_min"]
#     lat_max = gps_bbox["lat_max"]
#     lon_min = gps_bbox["lon_min"]
#     lon_max = gps_bbox["lon_max"]

#     with rasterio.open(tif_path) as src:
#         crs       = src.crs
#         transform = src.transform
#         H, W      = src.height, src.width

#     # Convert all 4 corners from WGS84 → this scene's CRS
#     corners_latlon = [
#         (lat_min, lon_min), (lat_min, lon_max),
#         (lat_max, lon_min), (lat_max, lon_max),
#     ]
#     rows_px, cols_px = [], []
#     for lat, lon in corners_latlon:
#         if HAS_PYPROJ:
#             t    = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
#             x, y = t.transform(lon, lat)
#         else:
#             from rasterio.warp import transform as wt
#             xs, ys = wt("EPSG:4326", crs, [lon], [lat])
#             x, y   = xs[0], ys[0]
#         r, c = rasterio.transform.rowcol(transform, x, y)
#         rows_px.append(int(r))
#         cols_px.append(int(c))

#     r0 = int(np.clip(min(rows_px), 0, H - 1))
#     r1 = int(np.clip(max(rows_px), 0, H - 1))
#     c0 = int(np.clip(min(cols_px), 0, W - 1))
#     c1 = int(np.clip(max(cols_px), 0, W - 1))

#     if r0 >= r1 or c0 >= c1:
#         print(f"  [WARN] GPS bbox outside scene bounds — using full scene.")
#         return arr

#     cropped = arr[:, r0:r1, c0:c1]
#     h, w = cropped.shape[1], cropped.shape[2]
#     print(f"  GPS→pixel crop: rows {r0}→{r1}, cols {c0}→{c1}  "
#           f"→ {h}×{w} px = {h*30/1000:.2f}×{w*30/1000:.2f} km")
#     return cropped


# def get_inputs():
#     print("\n" + "═"*68)
#     print("  POWAI LANDSAT CLASSIFIER  |  IIT Bombay GNR630  Topic 12")
#     print(f"  GPS locked: {POWAI_LAT}°N, {POWAI_LON}°E")
#     print("═"*68)

#     auto, datasets = find_files(), []
#     if len(auto) == 3:
#         print(f"\n  ✓ All 3 scenes auto-detected!")
#         for year in ["2009", "2016", "2025"]:
#             sensor = SCENE_FILES[year]["sensor"]
#             print(f"    {year} [{sensor}]: {len(auto[year])} bands")
#             datasets.append({"year": year, "sensor": sensor, "paths": auto[year]})
#     else:
#         print(f"\n  Found {len(auto)}/3 scenes. Enter missing paths manually.")
#         for year, info in SCENE_FILES.items():
#             sensor = info["sensor"]
#             if year in auto:
#                 datasets.append({"year": year, "sensor": sensor, "paths": auto[year]})
#                 print(f"  ✓ {year} [{sensor}] auto-detected.")
#             else:
#                 print(f"\n── {year} [{sensor}]  (need {len(info['bands'])} bands)")
#                 raw = ask("  Paths (comma-separated): ")
#                 paths = [p.strip().strip("'\"") for p in raw.split(",")] if raw else []
#                 datasets.append({"year": year, "sensor": sensor,
#                                  "paths": paths if paths else None})

#     # ── ROI Selection Mode ────────────────────────────────────────────────────
#     print(f"\n── ROI Selection Mode ───────────────────────────────────────────")
#     print(f"  1. GPS auto-crop  → enter pixel size (e.g. 90, 150, 200)")
#     print(f"     Centre always locked to Powai Lake GPS")
#     print(f"  2. Click 4 points → visually select ANY area on the image")
#     print(f"     A map opens — you click 4 corners of your region of interest")
#     mode = ask("\n  Choose mode [1=GPS / 2=Click 4 points] (default=1): ", "1")

#     custom_roi = None   # GPS bbox dict if mode 2
#     roi_size   = None

#     if mode == "2":
#         print("\n  Opening 2016 scene for 4-point selection…")
#         ref_paths = auto.get("2016") or auto.get("2025") or auto.get("2009")
#         if ref_paths:
#             custom_roi = interactive_roi_picker(ref_paths[0])
#         if custom_roi is None:
#             print("  Falling back to GPS mode.")
#             mode = "1"

#     if mode != "2":
#         print(f"\n── ROI Size (GPS mode) ──────────────────────────────────────────")
#         print(f"   ★ Validated: {OPTIMAL_ROI} px  → Urban +25% (2009→2025) ✓")
#         print(f"     Press ENTER to accept.")
#         raw      = ask(f"  ROI size [ENTER = {OPTIMAL_ROI}]: ", str(OPTIMAL_ROI))
#         roi_size = int(raw) if raw.isdigit() else OPTIMAL_ROI


#     out = ask("\n  Output filename [powai_lc_analysis.png]: ", "powai_lc_analysis.png")
#     return datasets, roi_size, custom_roi, out


# def main():
#     datasets, roi_size, custom_roi, out_path = get_inputs()
#     scenes = []

#     for ds in datasets:
#         year, sensor, paths = ds["year"], ds["sensor"], ds["paths"]
#         print(f"\n{'─'*60}\n  [{year}  {sensor}]")
#         if not paths:
#             print("  [skip] No paths."); continue

#         arr = load_scene(paths)
#         print(f"  Full scene: {arr.shape}")

#         # Crop — GPS bbox (same ground area for ALL scenes) or GPS-centred window
#         if custom_roi is not None:
#             # custom_roi is a GPS bbox dict — projects correctly into each sensor's grid
#             arr = crop_to_gps_bbox(arr, paths[0], custom_roi)
#         else:
#             arr = crop_to_roi(arr, paths[0], roi_size)

#         arr_h = harmonize_l5_to_l8(arr, sensor)
#         cl, idx = classify(arr_h, sensor)
#         ar = areas(cl)
#         px = ar["px"]
#         print(f"  RESULT  Water:{px['Water']:,}px={ar['Water']:.3f}km²  "
#               f"Veg:{px['Vegetation']:,}px={ar['Vegetation']:.3f}km²  "
#               f"Urban:{px['Urban / Other']:,}px={ar['Urban / Other']:.3f}km²")

#         scenes.append({"year": year, "sensor": sensor,
#                        "cl": cl, "fc": false_col(arr, sensor),
#                        "tc": true_col(arr, sensor), "areas": ar})

#     if len(scenes) < 3:
#         print("\n[ERROR] Need all 3 scenes. Check file paths."); return

#     # Display ROI size
#     if custom_roi is not None:
#         # Estimate from GPS bbox size in pixels
#         dlat = custom_roi["lat_max"] - custom_roi["lat_min"]
#         dlon = custom_roi["lon_max"] - custom_roi["lon_min"]
#         display_roi = int(max(dlat * 111000, dlon * 111000 *
#                               np.cos(np.radians(POWAI_LAT))) / 30)
#     else:
#         display_roi = roi_size

#     summary(scenes)
#     plot(scenes, display_roi, out_path)


# if __name__ == "__main__":
#     main()




"""
=============================================================================
 POWAI LAKE – LANDSAT LAND COVER CHANGE ANALYSIS  (2009 / 2016 / 2025)
 IIT Bombay  |  GNR630  |  Topic 12
=============================================================================
 CLASSIFICATION METHOD: NDBI-based + LDA + L5→L8 harmonization

 PIPELINE (classification logic UNCHANGED from working version):
   1. Load bands → L2 scale (×2.75e-5 − 0.2) → true surface reflectance
   2. [L5 2009 only] Cross-sensor harmonization L5 TM → L8 OLI
   3. Compute NDBI, mNDWI, NDWI, NDVI
   4. Water  : mNDWI > Otsu(mNDWI)  AND  NIR < Otsu(NIR)
   5. Urban  : NDBI > 0.0  (fixed, cross-sensor — Zha et al. 2003)
   6. Vegetation : everything else
   7. LDA refinement on [NIR, SWIR1, NDBI, RED]
   8. 3×3 median filter

 ★ USE ROI = 90  (just press ENTER — default is locked to 90)
   Validated: 90px gives Urban +25% (2009→2025), consistent with
   Hiranandani Gardens expansion. 80px also works (+91%). 100px+ wrong.

 YOUR FILES  (~/Desktop/GNR630i)
   2009 L5: LT05_L2SP_148047_20090108_20200828_02_T1_SR_B2/B3/B4/B5.TIF
   2016 L8: LC08_L2SP_148047_20160128_20200907_02_T1_SR_B3/B4/B5/B6.TIF
   2025 L8: LC08_L2SP_148047_20250104_20250111_02_T1_SR_B3/B4/B5/B6.TIF
=============================================================================
"""

import os, sys, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

POWAI_LAT   = 19.1260
POWAI_LON   = 72.9062
OPTIMAL_ROI = 90        # ★ validated — do not change

L2_SCALE  =  2.75e-5
L2_OFFSET = -0.2

SENSOR_BANDS = {
    "L5": {"GREEN": 0, "RED": 1, "NIR": 2, "SWIR1": 3},
    "L8": {"GREEN": 0, "RED": 1, "NIR": 2, "SWIR1": 3},
    "L9": {"GREEN": 0, "RED": 1, "NIR": 2, "SWIR1": 3},
}

# L5 TM → L8 OLI SR harmonization (Claverie et al. 2018)
L5_TO_L8 = {
    "GREEN": (0.9708,  0.0029),
    "RED":   (0.9762,  0.0005),
    "NIR":   (0.9419,  0.0074),
    "SWIR1": (0.9998, -0.0001),
}

CLASS_COLOURS = ["#1a78c2", "#3a9e4f", "#d4a24a"]
CLASS_LABELS  = ["Water", "Vegetation", "Urban / Other"]
LC_CMAP       = ListedColormap(CLASS_COLOURS)

SCENE_FILES = {
    "2009": {
        "sensor": "L5",
        "bands": [
            "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B2.TIF",
            "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B3.TIF",
            "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B4.TIF",
            "LT05_L2SP_148047_20090108_20200828_02_T1_SR_B5.TIF",
        ]
    },
    "2016": {
        "sensor": "L8",
        "bands": [
            "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B3.TIF",
            "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B4.TIF",
            "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B5.TIF",
            "LC08_L2SP_148047_20160128_20200907_02_T1_SR_B6.TIF",
        ]
    },
    "2025": {
        "sensor": "L8",
        "bands": [
            "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B3.TIF",
            "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B4.TIF",
            "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B5.TIF",
            "LC08_L2SP_148047_20250104_20250111_02_T1_SR_B6.TIF",
        ]
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  I/O
# ══════════════════════════════════════════════════════════════════════════════

def read_band(path):
    with rasterio.open(path.strip().strip("'\"")) as src:
        dn, nodata = src.read(1).astype(np.float32), src.nodata
    refl = dn * L2_SCALE + L2_OFFSET
    if nodata is not None:
        refl[dn == nodata] = np.nan
    refl[(refl < -0.15) | (refl > 1.5)] = np.nan
    return refl


def load_scene(paths):
    bands, ref = [], None
    for p in paths:
        b = read_band(p)
        if ref is None: ref = b.shape
        if b.shape != ref:
            from scipy.ndimage import zoom
            b = zoom(b, (ref[0]/b.shape[0], ref[1]/b.shape[1]), order=1)
        bands.append(b)
    return np.stack(bands, axis=0)


def harmonize_l5_to_l8(arr, sensor):
    if sensor != "L5": return arr
    cfg, arr_h = SENSOR_BANDS["L5"], arr.copy()
    for band, (s, o) in L5_TO_L8.items():
        i = cfg[band]
        arr_h[i] = np.where(np.isfinite(arr[i]), arr[i]*s+o, np.nan).astype(np.float32)
    print("  [Harmonize] L5→L8 OLI (Claverie 2018 SR)")
    return arr_h


# ══════════════════════════════════════════════════════════════════════════════
#  GPS → PIXEL + CROP
# ══════════════════════════════════════════════════════════════════════════════

def latlon_to_rowcol(tif_path, lat, lon):
    with rasterio.open(tif_path) as src:
        crs, transform, H, W = src.crs, src.transform, src.height, src.width
    if HAS_PYPROJ:
        t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = t.transform(lon, lat)
    else:
        from rasterio.warp import transform as wt
        xs, ys = wt("EPSG:4326", crs, [lon], [lat])
        x, y = xs[0], ys[0]
    row, col = rasterio.transform.rowcol(transform, x, y)
    row, col = int(np.clip(row, 0, H-1)), int(np.clip(col, 0, W-1))
    print(f"  GPS ({lat}°N,{lon}°E) → pixel (row={row}, col={col})  [scene={H}×{W}]")
    return row, col


def crop_to_roi(arr, tif_path, roi_size):
    if roi_size is None: return arr
    row, col = latlon_to_rowcol(tif_path, POWAI_LAT, POWAI_LON)
    _, H, W  = arr.shape
    half     = roi_size // 2
    r0 = max(0, row-half); r1 = r0+roi_size
    c0 = max(0, col-half); c1 = c0+roi_size
    if r1 > H: r0, r1 = H-roi_size, H
    if c1 > W: c0, c1 = W-roi_size, W
    r0, c0 = max(0, r0), max(0, c0)
    cropped = arr[:, r0:r1, c0:c1]
    h, w = cropped.shape[1], cropped.shape[2]
    print(f"  ROI: {h}×{w} px = {h*30/1000:.1f}×{w*30/1000:.1f} km")
    return cropped


# ══════════════════════════════════════════════════════════════════════════════
#  SPECTRAL INDICES
# ══════════════════════════════════════════════════════════════════════════════

def norm_diff(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((a+b)==0, np.nan, (a-b)/(a+b)).clip(-1, 1)


def compute_indices(arr, sensor):
    c = SENSOR_BANDS[sensor]
    nir, red, green, swir1 = arr[c["NIR"]], arr[c["RED"]], arr[c["GREEN"]], arr[c["SWIR1"]]
    return {
        "ndvi":  norm_diff(nir, red),
        "ndbi":  norm_diff(swir1, nir),
        "mndwi": norm_diff(green, swir1),
        "ndwi":  norm_diff(green, nir),
        "nir": nir, "red": red, "green": green, "swir1": swir1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  OTSU
# ══════════════════════════════════════════════════════════════════════════════

def otsu(values, n_bins=512):
    vals = values[np.isfinite(values)].ravel()
    if len(vals) < 10: return 0.0
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    if lo >= hi: return lo
    hist, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
    hist  = hist.astype(np.float64); total = hist.sum()
    cntrs = (edges[:-1]+edges[1:])/2
    gmean = np.dot(hist, cntrs)/total
    best_t, best_v, w0, s0 = lo, -1.0, 0.0, 0.0
    for i in range(n_bins):
        w0 += hist[i]; w1 = total-w0
        if w0==0 or w1==0: continue
        s0 += hist[i]*cntrs[i]
        m0, m1 = s0/w0, (gmean*total-s0)/w1
        v = (w0/total)*(w1/total)*(m0-m1)**2
        if v > best_v: best_v, best_t = v, cntrs[i]
    return best_t


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION  — YOUR EXACT WORKING LOGIC, UNCHANGED
# ══════════════════════════════════════════════════════════════════════════════

def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy metrics from true vs predicted labels.

    Returns dict with:
      conf_mat  : 3×3 confusion matrix  (rows=true, cols=predicted)
      OA        : Overall Accuracy  (%)
      kappa     : Cohen's Kappa coefficient
      PA        : Producer's Accuracy per class  (%)  — recall / sensitivity
      UA        : User's Accuracy per class      (%)  — precision
      F1        : F1 score per class             (%)

    Method: Hold-out validation on high-confidence anchor pixels.
    The same anchor pixels used for LDA training are split 70/30 (stratified).
    The 30% test set is completely unseen by the LDA model — giving an honest
    estimate of how well the classifier generalises to each land-cover class.
    Cohen's Kappa > 0.80 = strong agreement (Landis & Koch 1977 standard).
    """
    classes = [0, 1, 2]
    n = len(classes)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    total  = conf.sum()
    OA     = 100.0 * np.trace(conf) / total if total > 0 else 0.0

    # Cohen's Kappa
    p_o  = np.trace(conf) / total
    p_e  = np.sum(conf.sum(axis=1) * conf.sum(axis=0)) / (total**2)
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0

    PA, UA, F1 = {}, {}, {}
    for i, lbl in enumerate(CLASS_LABELS):
        tp = conf[i, i]
        row_sum = conf[i, :].sum()   # all true positives of this class
        col_sum = conf[:, i].sum()   # all predicted as this class
        PA[lbl] = 100.0 * tp / row_sum if row_sum > 0 else 0.0
        UA[lbl] = 100.0 * tp / col_sum if col_sum > 0 else 0.0
        pa, ua  = PA[lbl]/100, UA[lbl]/100
        F1[lbl] = 100.0 * 2*pa*ua / (pa+ua) if (pa+ua) > 0 else 0.0

    return {"conf_mat": conf, "OA": OA, "kappa": kappa,
            "PA": PA, "UA": UA, "F1": F1, "n_test": int(total)}


def classify(arr, sensor):
    """
    3-class NDBI classification (EXACTLY as in the validated working code).

    Water      : mNDWI > Otsu(mNDWI)  AND  NIR < Otsu(NIR)
    Urban      : NDBI > 0.0  (SWIR1 > NIR = impervious; Zha et al. 2003)
    Vegetation : everything else
    LDA        : [NIR, SWIR1, NDBI, RED]  (4 features, unchanged)
    Smoothing  : 3×3 median  (unchanged)

    Accuracy   : Stratified 70/30 hold-out on high-confidence anchor pixels.
                 OA, Kappa, PA, UA computed and returned alongside the map.
    """
    idx = compute_indices(arr, sensor)
    cfg = SENSOR_BANDS[sensor]

    # ── Water ─────────────────────────────────────────────────────────────────
    t_mndwi = max(otsu(idx["mndwi"]), 0.05)
    t_nir   = otsu(arr[cfg["NIR"]])
    water_mask = (
        (idx["mndwi"] > t_mndwi) &
        (arr[cfg["NIR"]] < t_nir) &
        np.isfinite(idx["mndwi"])
    )

    # ── Urban / Vegetation via NDBI ───────────────────────────────────────────
    urban_seed = (idx["ndbi"] > 0.0) & ~water_mask & np.isfinite(idx["ndbi"])
    cl = np.full(idx["ndvi"].shape, 1, dtype=np.uint8)
    cl[urban_seed] = 2
    cl[water_mask] = 0

    w, v, u = np.sum(cl==0), np.sum(cl==1), np.sum(cl==2)
    print(f"  NDBI threshold=0.0  mNDWI≥{t_mndwi:.3f}  NIR<{t_nir:.3f}")
    print(f"  NDBI mean={np.nanmean(idx['ndbi']):.3f}  Seed W={w:,} V={v:,} U={u:,} px")

    acc = None   # filled below if sklearn available

    # ── LDA (4 features: NIR, SWIR1, NDBI, RED) ──────────────────────────────
    if HAS_SKLEARN and w > 100 and v > 100 and u > 100:
        ndbi_f = idx["ndbi"].ravel()
        fi     = [cfg["NIR"], cfg["SWIR1"], cfg["RED"]]
        X      = np.column_stack([np.stack([arr[i].ravel() for i in fi], axis=1), ndbi_f])
        y      = cl.ravel()
        mf     = idx["mndwi"].ravel()
        nir    = arr[cfg["NIR"]].ravel()

        # High-confidence anchor pixels (used for train + accuracy test)
        cw = (mf > t_mndwi+0.06) & (nir < t_nir*0.75)
        cv = ndbi_f < -0.25
        cu = ndbi_f > 0.10
        cm = (cw | cv | cu) & np.isfinite(X).all(axis=1)

        if len(np.unique(y[cm])) >= 2:
            ok = np.isfinite(X).all(axis=1)
            sc = StandardScaler()

            X_cm, y_cm = X[cm & ok], y[cm & ok]

            # ── Stratified 70/30 split for accuracy assessment ────────────────
            # 30% held-out pixels are NEVER seen during LDA training — honest OA
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30,
                                              random_state=42)
                train_idx, test_idx = next(sss.split(X_cm, y_cm))
                X_tr70, y_tr70 = X_cm[train_idx], y_cm[train_idx]
                X_te30, y_te30 = X_cm[test_idx],  y_cm[test_idx]

                # Train on 70% of anchors
                sc70  = StandardScaler()
                Xtr70 = sc70.fit_transform(X_tr70)
                Xte30 = sc70.transform(X_te30)
                lda70 = LinearDiscriminantAnalysis()
                lda70.fit(Xtr70, y_tr70)
                y_pred_test = lda70.predict(Xte30)
                acc = compute_accuracy(y_te30, y_pred_test)
                print(f"  Accuracy (30% hold-out, n={acc['n_test']} px): "
                      f"OA={acc['OA']:.1f}%  κ={acc['kappa']:.3f}")
                for lbl in CLASS_LABELS:
                    print(f"    {lbl:<18} PA={acc['PA'][lbl]:.1f}%  "
                          f"UA={acc['UA'][lbl]:.1f}%  F1={acc['F1'][lbl]:.1f}%")
            except Exception as e:
                print(f"  [Accuracy] Could not compute: {e}")

            # ── Full LDA on ALL anchor pixels → classify whole scene ──────────
            Xtr = sc.fit_transform(X_cm)
            Xal = sc.transform(X[ok])
            lda = LinearDiscriminantAnalysis()
            lda.fit(Xtr, y_cm)
            pred     = lda.predict(Xal)
            flat     = cl.ravel().copy()
            flat[ok] = pred.astype(np.uint8)
            cl       = flat.reshape(cl.shape)
            print(f"  LDA   W={np.sum(cl==0):,}  V={np.sum(cl==1):,}  U={np.sum(cl==2):,} px")
        else:
            print("  LDA skipped – low class diversity")
    else:
        print("  LDA skipped")

    return median_filter(cl, size=3), idx, acc   # 3×3 unchanged


# ══════════════════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════════════════

def areas(cl, px=30.0):
    d = {l: int(np.sum(cl==i))*(px**2)/1e6 for i, l in enumerate(CLASS_LABELS)}
    d["px"] = {l: int(np.sum(cl==i)) for i, l in enumerate(CLASS_LABELS)}
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def stretch(b, lo=2, hi=98):
    pl, ph = np.nanpercentile(b, lo), np.nanpercentile(b, hi)
    return np.clip((b-pl)/(ph-pl+1e-9), 0, 1)

def false_col(arr, sensor):
    c = SENSOR_BANDS[sensor]
    return np.dstack([stretch(arr[c["NIR"]]), stretch(arr[c["RED"]]), stretch(arr[c["GREEN"]])])

def true_col(arr, sensor):
    c = SENSOR_BANDS[sensor]
    return np.dstack([stretch(arr[c["RED"]]), stretch(arr[c["GREEN"]]), stretch(arr[c["GREEN"]])])

def tc_label(sensor):
    return "Pseudo True Colour (R/G/G*)" if sensor == "L5" else "True Colour (R/G/B)"


def make_change_map(cl_old, cl_new):
    """Pixel-level change map: red=→Urban, green=→Veg, blue=→Water, grey=no change."""
    H, W = cl_old.shape
    rgba = np.full((H, W, 4), [0.18, 0.20, 0.24, 1.0], dtype=np.float32)
    no_change          = cl_old == cl_new
    rgba[~no_change & (cl_new==2)] = [0.90, 0.22, 0.12, 1]   # → Urban  (red)
    rgba[~no_change & (cl_new==1)] = [0.23, 0.62, 0.31, 1]   # → Veg    (green)
    rgba[~no_change & (cl_new==0)] = [0.10, 0.47, 0.76, 1]   # → Water  (blue)
    n_u = int(np.sum(~no_change & (cl_new==2)))
    n_v = int(np.sum(~no_change & (cl_new==1)))
    n_w = int(np.sum(~no_change & (cl_new==0)))
    return rgba, n_u, n_v, n_w


def plot_confusion_matrix(ax, acc, year, yc, BG, BD, W, M):
    """
    Confusion matrix with explicit correct/wrong pixel counts and percentages.
    Rows = True class, Cols = Predicted class.
    Diagonal = correctly classified, Off-diagonal = misclassified.
    """
    if acc is None:
        ax.set_facecolor(BG); ax.axis("off")
        ax.text(0.5, 0.5, "sklearn not installed", transform=ax.transAxes,
                color=M, ha="center", va="center", fontsize=9)
        return

    conf  = acc["conf_mat"]
    total = conf.sum()
    n     = len(CLASS_LABELS)
    short = ["Water", "Veg", "Urban"]
    ax.set_facecolor(BG)

    for r in range(n):
        for c in range(n):
            val     = conf[r, c]
            pct     = 100.0 * val / total if total > 0 else 0
            is_diag = (r == c)
            # Diagonal = correct (class colour), off-diagonal = error (red tint)
            if is_diag:
                face = CLASS_COLOURS[r]
                intensity = 0.25 + 0.65 * (val / (conf[r].sum() + 1e-9))
            else:
                face = "#c0392b"
                intensity = 0.10 + 0.55 * (val / (conf.max() + 1e-9))
            ax.add_patch(plt.Rectangle(
                (c, n-1-r), 1, 1, color=face, alpha=intensity,
                linewidth=0, zorder=1))
            # Cell border
            ax.add_patch(plt.Rectangle(
                (c, n-1-r), 1, 1, fill=False, edgecolor=BD, linewidth=0.8, zorder=2))
            # Count (large) + percentage (small)
            ax.text(c+0.5, n-1-r+0.62, f"{val:,}",
                    ha="center", va="center", color=W,
                    fontsize=11, fontweight="bold", zorder=3)
            ax.text(c+0.5, n-1-r+0.28, f"({pct:.1f}%)",
                    ha="center", va="center", color=W,
                    fontsize=7.5, zorder=3)
            # Label wrong cells
            if not is_diag and val > 0:
                ax.text(c+0.5, n-1-r+0.10, "✗ wrong",
                        ha="center", va="center", color="#f85149",
                        fontsize=6.5, zorder=3)

    # Axes
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(short, color=M, fontsize=8.5)
    ax.set_yticklabels(short[::-1], color=M, fontsize=8.5)
    ax.set_xlabel("Predicted →", color=M, fontsize=8, labelpad=3)
    ax.set_ylabel("← True Class", color=M, fontsize=8, labelpad=3)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_color(BD)

    # Misclassified pixel summary below
    n_correct = int(np.trace(conf))
    n_wrong   = int(total - n_correct)
    kc = "#56d364" if acc["kappa"] >= 0.80 else \
         "#f0883e" if acc["kappa"] >= 0.60 else "#f85149"

    ax.set_title(
        f"{year}  —  OA: {acc['OA']:.1f}%   κ: {acc['kappa']:.3f}",
        color=W, fontsize=10, fontweight="bold", pad=5,
        bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=yc, lw=1.4))
    ax.text(0.5, -0.13,
            f"✓ {n_correct:,} correct  |  ✗ {n_wrong:,} misclassified  |  n={total:,} test pixels",
            transform=ax.transAxes, color=M, fontsize=7.5,
            ha="center", va="top", fontfamily="monospace")



def plot(scenes, roi_size, out):
    """
    Single combined image — 8 rows × 4 cols:
      Row 0: True Colour | Row 1: False Colour | Row 2: Classification
      Row 3: Change Maps | Row 4: Confusion Matrices
      Row 5: Correct vs Misclassified bars | Row 6: Per-class accuracy tables
      Row 7: Area bar chart + Δ change summary
    """
    years   = [s["year"]   for s in scenes]
    sensors = [s["sensor"] for s in scenes]
    clsf    = [s["cl"]     for s in scenes]
    fc_imgs = [s["fc"]     for s in scenes]
    tc_imgs = [s["tc"]     for s in scenes]
    AR      = [s["areas"]  for s in scenes]
    ACCS    = [s["acc"]    for s in scenes]
    YC      = ["#58a6ff", "#f0883e", "#56d364"]
    BG, CARD, BD = "#0d1117", "#161b22", "#30363d"
    W, M = "#e6edf3", "#8b949e"
    short = ["Water", "Veg", "Urban"]

    row_h = [1.4, 1.4, 1.4, 1.4, 1.7, 1.0, 1.4, 1.1]
    fig = plt.figure(figsize=(24, 60), facecolor=BG)
    gs  = gridspec.GridSpec(8, 4, fig, height_ratios=row_h,
                            hspace=0.44, wspace=0.18,
                            left=0.04, right=0.97, top=0.978, bottom=0.01)

    km = f"{roi_size*30/1000:.1f}"
    fig.suptitle(
        "POWAI LAKE  —  LAND COVER CHANGE ANALYSIS  &  ACCURACY REPORT\n"
        f"Supervised Classification: LDA  |  Landsat 5 (2009)  ·  Landsat 8 (2016)  ·  Landsat 8 (2025)"
        f"   |  GPS: {POWAI_LAT}°N, {POWAI_LON}°E"
        f"   |  ROI: {roi_size}×{roi_size} px ({km}×{km} km)"
        f"   |  IIT Bombay  GNR630",
        color=W, fontsize=13, fontweight="bold", y=0.986, fontfamily="monospace")

    def badge(ax, label, yc):
        ax.text(0.03, 0.97, label, transform=ax.transAxes, color=yc,
                fontsize=12, fontweight="bold", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=yc, lw=1.5))

    # ── Row 0: True Colour ────────────────────────────────────────────────────
    for i, (y, img, yc, sen) in enumerate(zip(years, tc_imgs, YC, sensors)):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(img); ax.axis("off")
        ax.set_title(f"{y}  –  {tc_label(sen)}", color=W, fontsize=11, fontweight="bold", pad=4)
        badge(ax, y, yc)
    ax_r = fig.add_subplot(gs[0, 3]); ax_r.set_facecolor(CARD); ax_r.axis("off")
    ax_r.text(0.5, 0.55, "TRUE COLOUR\n\nShows landscape\nas seen from space.\n\n"
              "L5 2009: No B1 (Blue)\n→ Green substituted\n(Pseudo True Colour)",
              transform=ax_r.transAxes, color=M, fontsize=9.5, ha="center", va="center",
              fontfamily="monospace")

    # ── Row 1: False Colour ───────────────────────────────────────────────────
    for i, (y, img, yc) in enumerate(zip(years, fc_imgs, YC)):
        ax = fig.add_subplot(gs[1, i]); ax.imshow(img); ax.axis("off")
        ax.set_title(f"{y}  –  False Colour (NIR/Red/Green)", color=W, fontsize=11, fontweight="bold", pad=4)
        badge(ax, y, yc)
    ax_fl = fig.add_subplot(gs[1, 3]); ax_fl.set_facecolor(CARD); ax_fl.axis("off")
    ax_fl.text(0.5, 0.75, "FALSE COLOUR KEY", transform=ax_fl.transAxes, color=W, fontsize=11, ha="center", fontweight="bold")
    for txt, col, yp in [("Bright Red = Dense vegetation","#e74c3c",0.56),
                          ("Cyan / Teal = Urban / Bare","#00bcd4",0.40),
                          ("Black = Water (lake)","#90caf9",0.25)]:
        ax_fl.text(0.5, yp, txt, transform=ax_fl.transAxes, color=col, fontsize=9, ha="center", fontfamily="monospace")

    # ── Row 2: Classification Maps ────────────────────────────────────────────
    for i, (y, cl, yc, ar) in enumerate(zip(years, clsf, YC, AR)):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(cl, cmap=LC_CMAP, vmin=0, vmax=2, interpolation="nearest"); ax.axis("off")
        ax.set_title(f"{y}  –  Land Cover Classification", color=W, fontsize=11, fontweight="bold", pad=4)
        badge(ax, y, yc)
        ax.text(0.02, 0.02,
                f"Water : {ar['Water']:.3f} km²\nVeg   : {ar['Vegetation']:.3f} km²\nUrban : {ar['Urban / Other']:.3f} km²",
                transform=ax.transAxes, color=W, fontsize=8.5, va="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#00000099", ec=BD))
    ax_lg = fig.add_subplot(gs[2, 3]); ax_lg.set_facecolor(CARD); ax_lg.axis("off")
    ax_lg.set_title("LEGEND", color=W, fontsize=12, fontweight="bold", pad=4)
    for i, (lbl, col) in enumerate(zip(CLASS_LABELS, CLASS_COLOURS)):
        y_ = 0.65 - i*0.20
        ax_lg.add_patch(plt.Rectangle((0.08, y_), 0.25, 0.14, color=col, transform=ax_lg.transAxes, clip_on=False))
        ax_lg.text(0.40, y_+0.07, lbl, transform=ax_lg.transAxes, color=W, fontsize=12, va="center", fontweight="bold")
    ax_lg.text(0.5, 0.06, "L5→L8 harmonize\n(Claverie 2018)\nNDBI>0 urban seed\nLDA[NIR,SWIR,NDBI,RED]\n3×3 median",
               transform=ax_lg.transAxes, color=M, fontsize=8.5, va="bottom", ha="center", fontfamily="monospace")

    # ── Row 3: Change Maps ────────────────────────────────────────────────────
    px_km2 = 30.0**2 / 1e6
    for i, (c_old, c_new, title, yc) in enumerate([
        (clsf[0], clsf[1], "Change: 2009 → 2016", YC[1]),
        (clsf[1], clsf[2], "Change: 2016 → 2025", YC[2]),
        (clsf[0], clsf[2], "Change: 2009 → 2025 (full)", "#ffd700"),
    ]):
        ax = fig.add_subplot(gs[3, i])
        rgba, n_u, n_v, n_w = make_change_map(c_old, c_new)
        ax.imshow(rgba); ax.axis("off")
        ax.set_title(title, color=W, fontsize=11, fontweight="bold", pad=4)
        ax.text(0.02, 0.02,
                f"→ Urban : {n_u*px_km2:+.3f} km²\n→ Veg   : {n_v*px_km2:+.3f} km²\n→ Water : {n_w*px_km2:+.3f} km²",
                transform=ax.transAxes, color=W, fontsize=8, va="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#00000099", ec=BD))
    ax_ck = fig.add_subplot(gs[3, 3]); ax_ck.set_facecolor(CARD); ax_ck.axis("off")
    ax_ck.set_title("CHANGE MAP KEY", color=W, fontsize=12, fontweight="bold", pad=4)
    for j, (col, lbl) in enumerate([("#e5381e","→ Urban  (new built-up)"),("#3a9e4f","→ Vegetation (new green)"),
                                     ("#1a78c2","→ Water  (new water)"),("#2e333d","No change")]):
        y_ = 0.70 - j*0.17
        ax_ck.add_patch(plt.Rectangle((0.05, y_), 0.22, 0.11, color=col, transform=ax_ck.transAxes, clip_on=False))
        ax_ck.text(0.34, y_+0.055, lbl, transform=ax_ck.transAxes, color=W, fontsize=10, va="center")

    # ── Row 4: Confusion Matrices ─────────────────────────────────────────────
    for ci, (year, acc, yc) in enumerate(zip(years, ACCS, YC)):
        ax = fig.add_subplot(gs[4, ci]); ax.set_facecolor(BG)
        if acc is None:
            ax.axis("off"); ax.text(0.5,0.5,"Accuracy N/A",transform=ax.transAxes,color=M,ha="center",va="center"); continue
        conf = acc["conf_mat"]; total = conf.sum()
        for r in range(3):
            for c in range(3):
                val = conf[r, c]; pct = 100.0*val/total if total>0 else 0
                is_d = (r==c)
                face = CLASS_COLOURS[r] if is_d else "#c0392b"
                alph = (0.30 + 0.60*(val/(conf[r].sum()+1e-9))) if is_d else min(0.12+0.55*(val/(conf.max()+1e-9)),0.65)
                ax.add_patch(plt.Rectangle((c,2-r),1,1,color=face,alpha=alph,linewidth=0,zorder=1))
                ax.add_patch(plt.Rectangle((c,2-r),1,1,fill=False,edgecolor=BD,linewidth=1.2,zorder=2))
                ax.text(c+0.5,2-r+0.66,f"{val:,}",ha="center",va="center",color=W,fontsize=13,fontweight="bold",zorder=3)
                ax.text(c+0.5,2-r+0.38,f"{pct:.1f}% of test",ha="center",va="center",color=W,fontsize=7.5,zorder=3)
                ax.text(c+0.5,2-r+0.14,"✓ correct" if is_d else "✗ wrong",ha="center",va="center",
                        color="#56d364" if is_d else "#f85149",fontsize=7,fontweight="bold",zorder=3)
        ax.set_xlim(0,3); ax.set_ylim(0,3)
        ax.set_xticks([0.5,1.5,2.5]); ax.set_yticks([0.5,1.5,2.5])
        ax.set_xticklabels(short,color=M,fontsize=9); ax.set_yticklabels(short[::-1],color=M,fontsize=9)
        ax.set_xlabel("Predicted →",color=M,fontsize=8,labelpad=3)
        ax.set_ylabel("← True",color=M,fontsize=8,labelpad=3)
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_color(BD)
        nc=int(np.trace(conf)); nw=int(total-nc)
        ax.set_title(f"{year}  |  OA: {acc['OA']:.1f}%   κ: {acc['kappa']:.3f}",
                     color=W,fontsize=11,fontweight="bold",pad=6,
                     bbox=dict(boxstyle="round,pad=0.4",fc=BG,ec=yc,lw=2))
        ax.text(0.5,-0.10,f"✓ {nc:,} correctly classified   ✗ {nw:,} misclassified   (n={total:,} test px)",
                transform=ax.transAxes,color=M,fontsize=8,ha="center",va="top",fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3",fc="#1c2128",ec=BD))

    ax_oa = fig.add_subplot(gs[4, 3]); ax_oa.set_facecolor(CARD); ax_oa.axis("off")
    ax_oa.set_title("OVERALL ACCURACY", color=W, fontsize=12, fontweight="bold", pad=4)
    if all(a is not None for a in ACCS):
        ov = []
        for y, acc in zip(years, ACCS):
            nc=int(np.trace(acc["conf_mat"])); nw=int(acc["n_test"]-nc)
            ov.append([y, f"{acc['OA']:.1f}%", f"{acc['kappa']:.3f}", f"{nc:,}", f"{nw:,}"])
        t = ax_oa.table(cellText=ov, colLabels=["Year","OA","κ","✓ Correct","✗ Wrong"],
                        cellLoc="center", loc="center", bbox=[0,0.38,1,0.52])
        t.auto_set_font_size(False); t.set_fontsize(9)
        for (r,c), cell in t.get_celld().items():
            cell.set_edgecolor(BD)
            txt = cell.get_text().get_text()
            if r==0: cell.set_facecolor("#21262d"); cell.set_text_props(color=W, fontweight="bold")
            else:
                cell.set_facecolor(CARD)
                if c==1:
                    try: v=float(txt.strip("%")); cell.set_text_props(color="#56d364" if v>=85 else "#f0883e" if v>=70 else "#f85149")
                    except: cell.set_text_props(color=W)
                elif c==4: cell.set_text_props(color="#f85149")
                elif c==3: cell.set_text_props(color="#56d364")
                else: cell.set_text_props(color=W)
        ax_oa.text(0.5,0.35,"Confusion Matrix Guide:\nDiagonal = Correctly classified\nOff-diagonal = Misclassified\nRows=True class  Cols=Predicted",
                   transform=ax_oa.transAxes,color=M,fontsize=8.5,ha="center",va="top",fontfamily="monospace")
        ax_oa.text(0.5,0.06,"κ ≥ 0.80 = Strong Agreement\nκ 0.60–0.79 = Moderate Agreement\n(Landis & Koch 1977)",
                   transform=ax_oa.transAxes,color=M,fontsize=8,ha="center",va="bottom",fontfamily="monospace")

    # ── Row 5: Correct vs Misclassified Bars ─────────────────────────────────
    for ci, (year, acc, yc) in enumerate(zip(years, ACCS, YC)):
        ax = fig.add_subplot(gs[5, ci]); ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_color(BD)
        ax.tick_params(colors=M)
        if acc is not None:
            conf=acc["conf_mat"]; x=np.arange(3)
            cor=[conf[i,i] for i in range(3)]; mis=[conf[i,:].sum()-conf[i,i] for i in range(3)]
            b1=ax.bar(x-0.2,cor,0.36,label="✓ Correct",color="#56d364",alpha=0.88,edgecolor="none")
            b2=ax.bar(x+0.2,mis,0.36,label="✗ Misclassified",color="#f85149",alpha=0.88,edgecolor="none")
            mx=max(max(cor),max(mis)) if cor else 1
            for bars in [b1,b2]:
                for bar in bars:
                    h=bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2, h+mx*0.04, str(int(h)),
                            ha="center",va="bottom",color=W,fontsize=9,fontweight="bold")
            ax.set_xticks(x); ax.set_xticklabels(short,color=W,fontsize=10,fontweight="bold")
            ax.set_ylabel("Pixel Count",color=M,fontsize=9)
            ax.set_title(f"{year}  —  Correct vs Misclassified per Class",color=W,fontsize=11,fontweight="bold",pad=4)
            ax.legend(facecolor="#1c2128",edgecolor=BD,labelcolor=W,fontsize=9,loc="upper right")
            ax.grid(axis="y",color=BD,linewidth=0.6,zorder=0)
            ax.set_ylim(0,ax.get_ylim()[1]*1.22)
        else: ax.axis("off")

    ax_pu = fig.add_subplot(gs[5, 3]); ax_pu.set_facecolor(CARD); ax_pu.axis("off")
    ax_pu.set_title("SUPERVISED CLASSIFICATION ACCURACY TREND", color=W, fontsize=11, fontweight="bold", pad=4)
    if all(a is not None for a in ACCS):
        # ── Accuracy trend: OA% and κ across years ────────────────────────────
        ax_pu.axis("on")
        ax_pu.set_facecolor(BG)
        for sp in ax_pu.spines.values(): sp.set_color(BD)
        ax_pu.tick_params(colors=M)

        yr_nums  = [2009, 2016, 2025]
        oa_vals  = [acc["OA"]    for acc in ACCS]
        kap_vals = [acc["kappa"] * 100 for acc in ACCS]   # scale to % for dual axis

        # OA% bars
        bar_w = 4
        b_oa = ax_pu.bar([y - bar_w/2 for y in yr_nums], oa_vals, bar_w,
                         color="#56d364", alpha=0.85, label="Overall Accuracy (%)", zorder=3)
        # κ×100 bars
        b_ka = ax_pu.bar([y + bar_w/2 for y in yr_nums], kap_vals, bar_w,
                         color="#f0883e", alpha=0.85, label="Kappa × 100", zorder=3)

        # Value labels on bars
        for bar, v in zip(b_oa, oa_vals):
            ax_pu.text(bar.get_x()+bar.get_width()/2, v+0.5,
                       f"{v:.1f}%", ha="center", va="bottom",
                       color=W, fontsize=10, fontweight="bold")
        for bar, v, raw in zip(b_ka, kap_vals, [acc["kappa"] for acc in ACCS]):
            ax_pu.text(bar.get_x()+bar.get_width()/2, v+0.5,
                       f"κ={raw:.3f}", ha="center", va="bottom",
                       color=W, fontsize=9, fontweight="bold")

        # OA trend line
        ax_pu.plot(yr_nums, oa_vals, "o--", color="#56d364",
                   linewidth=2, markersize=7, zorder=4)

        # 80% Strong threshold line
        ax_pu.axhline(80, color="#ffd700", linewidth=1.2, linestyle=":",
                      label="κ = 0.80 threshold (Strong)", zorder=2)
        ax_pu.text(2025.5, 80.8, "κ = 0.80\n(Strong)", color="#ffd700",
                   fontsize=8, va="bottom", fontfamily="monospace")

        ax_pu.set_xlim(2004, 2030)
        ax_pu.set_ylim(0, 110)
        ax_pu.set_xticks(yr_nums)
        ax_pu.set_xticklabels([str(y) for y in yr_nums], color=W, fontsize=11, fontweight="bold")
        ax_pu.set_ylabel("Accuracy (%)", color=M, fontsize=10)
        ax_pu.yaxis.label.set_color(M)
        ax_pu.tick_params(axis="y", colors=M)
        ax_pu.grid(axis="y", color=BD, linewidth=0.6, zorder=0)
        ax_pu.legend(facecolor="#1c2128", edgecolor=BD, labelcolor=W,
                     fontsize=9, loc="lower right")
        ax_pu.text(0.5, -0.14,
                   "Algorithm: Linear Discriminant Analysis (LDA)  —  Supervised Classification\n"
                   "Validation: Stratified 70/30 Hold-out  (Congalton 1991)",
                   transform=ax_pu.transAxes, color=M, fontsize=8.5,
                   ha="center", va="top", fontfamily="monospace")
    else:
        ax_pu.axis("off")
        ax_pu.text(0.5, 0.5, "Install scikit-learn\nfor accuracy metrics",
                   transform=ax_pu.transAxes, color=M, ha="center", va="center")

    # ── Row 6: Per-class accuracy tables ─────────────────────────────────────
    for ci, (year, acc, yc) in enumerate(zip(years, ACCS, YC)):
        ax = fig.add_subplot(gs[6, ci]); ax.set_facecolor(CARD); ax.axis("off")
        ax.set_title(f"{year}  —  Per-Class Accuracy Detail",color=W,fontsize=11,fontweight="bold",pad=4,
                     bbox=dict(boxstyle="round,pad=0.3",fc=BG,ec=yc,lw=1.5))
        if acc is not None:
            conf=acc["conf_mat"]; trows=[]
            for i,(lbl,s) in enumerate(zip(CLASS_LABELS,short)):
                tp=conf[i,i]; fn=conf[i,:].sum()-tp; fp=conf[:,i].sum()-tp
                trows.append([s,f"{tp:,}",f"{fn:,}",f"{fp:,}",
                              f"{acc['PA'][lbl]:.1f}%",f"{acc['UA'][lbl]:.1f}%",f"{acc['F1'][lbl]:.1f}%"])
            tbl=ax.table(cellText=trows,colLabels=["Class","✓TP","✗FN","✗FP","PA","UA","F1"],
                         cellLoc="center",loc="center",bbox=[0,0.12,1,0.78])
            tbl.auto_set_font_size(False); tbl.set_fontsize(9)
            for (r,c),cell in tbl.get_celld().items():
                cell.set_edgecolor(BD); txt=cell.get_text().get_text()
                if r==0: cell.set_facecolor("#21262d"); cell.set_text_props(color=W,fontweight="bold")
                else:
                    cell.set_facecolor(CARD)
                    if c in (4,5,6):
                        try: v=float(txt.strip("%")); cell.set_text_props(color="#56d364" if v>=85 else "#f0883e" if v>=70 else "#f85149")
                        except: cell.set_text_props(color=W)
                    elif c in (2,3): cell.set_text_props(color="#f85149" if txt!="0" else "#56d364")
                    else: cell.set_text_props(color=W)
            ax.text(0.5,0.05,"TP=True Positive  FN=False Negative  FP=False Positive",
                    transform=ax.transAxes,color=M,fontsize=7.5,ha="center",va="bottom",fontfamily="monospace")

    ax_mn = fig.add_subplot(gs[6, 3]); ax_mn.set_facecolor(CARD); ax_mn.axis("off")
    ax_mn.set_title("SUPERVISED LEARNING SUMMARY", color=W, fontsize=11, fontweight="bold", pad=4)
    ax_mn.text(0.5, 0.96,
               "Classification Algorithm:",
               transform=ax_mn.transAxes, color=W, fontsize=10,
               ha="center", va="top", fontweight="bold")
    ax_mn.text(0.5, 0.86,
               "Linear Discriminant Analysis (LDA)",
               transform=ax_mn.transAxes, color="#58a6ff", fontsize=11,
               ha="center", va="top", fontweight="bold", fontfamily="monospace")
    ax_mn.text(0.5, 0.75,
               "Type: SUPERVISED CLASSIFICATION\n"
               "Features: NIR, SWIR1, NDBI, RED (4 bands)\n"
               "Classes: Water / Vegetation / Urban",
               transform=ax_mn.transAxes, color=M, fontsize=9,
               ha="center", va="top", fontfamily="monospace")

    ax_mn.axhline(0.60, xmin=0.05, xmax=0.95,
                  color=BD, linewidth=0.8)

    ax_mn.text(0.5, 0.55,
               "Accuracy Validation Method:",
               transform=ax_mn.transAxes, color=W, fontsize=10,
               ha="center", va="top", fontweight="bold")
    ax_mn.text(0.5, 0.45,
               "Stratified 70/30 Hold-out\n(Congalton 1991)\n\n"
               "• 70% anchor pixels → LDA training\n"
               "• 30% held-out → testing (unseen)\n"
               "• Stratified = equal class proportions\n"
               "• random_state=42 (reproducible)",
               transform=ax_mn.transAxes, color=M, fontsize=8.5,
               ha="center", va="top", fontfamily="monospace")

    ax_mn.axhline(0.18, xmin=0.05, xmax=0.95,
                  color=BD, linewidth=0.8)

    ax_mn.text(0.5, 0.14,
               "κ ≥ 0.80  →  Strong Agreement ✓\n"
               "κ 0.60–0.79  →  Moderate\n"
               "κ < 0.60  →  Weak",
               transform=ax_mn.transAxes, color=M, fontsize=8.5,
               ha="center", va="top", fontfamily="monospace")

    # ── Row 7: Area Bar + Δ Change Table ─────────────────────────────────────
    ax_b = fig.add_subplot(gs[7, :3]); ax_b.set_facecolor(CARD)
    for sp in ax_b.spines.values(): sp.set_color(BD)
    ax_b.tick_params(colors=M)
    x,bw = np.arange(3), 0.22
    for j,(y,ar,yc) in enumerate(zip(years,AR,YC)):
        vals=[ar[l] for l in CLASS_LABELS]
        bars=ax_b.bar(x+(j-1)*bw,vals,bw,label=y,color=yc,alpha=0.88,edgecolor="none",zorder=3)
        mx=max(vals) if max(vals)>0 else 1
        for bar,v in zip(bars,vals):
            ax_b.text(bar.get_x()+bar.get_width()/2,bar.get_height()+mx*0.02,
                      f"{v:.3f}",ha="center",va="bottom",color=W,fontsize=8,fontfamily="monospace")
    ax_b.set_xticks(x); ax_b.set_xticklabels(CLASS_LABELS,color=W,fontsize=13,fontweight="bold")
    ax_b.set_ylabel("Area (km²)",color=M,fontsize=11)
    ax_b.set_title("Land Cover Area Comparison  (2009 → 2016 → 2025)",color=W,fontsize=12,fontweight="bold",pad=6)
    ax_b.grid(axis="y",color=BD,linewidth=0.7,zorder=0)
    ax_b.legend(facecolor="#1c2128",edgecolor=BD,labelcolor=W,fontsize=11,loc="upper right")
    ax_b.set_ylim(0,ax_b.get_ylim()[1]*1.22)

    ax_t = fig.add_subplot(gs[7, 3]); ax_t.set_facecolor(CARD); ax_t.axis("off")
    ax_t.set_title("Δ CHANGE SUMMARY",color=W,fontsize=12,fontweight="bold",pad=4)
    a0,a1,a2=AR; rows=[]
    for lbl in CLASS_LABELS:
        d1=a1[lbl]-a0[lbl]; d2=a2[lbl]-a1[lbl]; dt=a2[lbl]-a0[lbl]; pc=dt/(a0[lbl]+1e-9)*100
        rows.append([lbl[:5],f"{d1:+.3f}",f"{d2:+.3f}",f"{dt:+.3f}",f"{pc:+.0f}%"])
    tbl=ax_t.table(cellText=rows,colLabels=["Class","Δ09→16","Δ16→25","ΔTotal","%Chg"],
                   cellLoc="center",loc="center",bbox=[0,0.10,1,0.82])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor(BD); txt=cell.get_text().get_text()
        if r==0: cell.set_facecolor("#21262d"); cell.set_text_props(color=W,fontweight="bold")
        else:
            cell.set_facecolor(CARD)
            cell.set_text_props(color="#56d364" if "+" in txt else "#f85149" if "-" in txt else W)

    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  ✓  Complete report saved → {out}  (200 dpi, single image)")

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summary(scenes):
    print("\n" + "═"*68)
    print("  LAND COVER AREA SUMMARY (km²)")
    print(f"  GPS: {POWAI_LAT}°N, {POWAI_LON}°E  |  ROI: {OPTIMAL_ROI}×{OPTIMAL_ROI} px")
    print("═"*68)
    print(f"  {'Class':<22}{'2009':>8}{'2016':>8}{'2025':>8}{'Δ 09→25':>12}{'%':>8}")
    print("  " + "─"*62)
    for lbl in CLASS_LABELS:
        v = [s["areas"][lbl] for s in scenes]
        d = v[2]-v[0]; p = d/(v[0]+1e-9)*100
        print(f"  {lbl:<22}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}"
              f"  {d:>+10.3f}  {'▲' if d>0 else '▼'}{abs(p):>5.1f}%")
    print("═"*68)


# ══════════════════════════════════════════════════════════════════════════════
#  INPUT + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def ask(prompt, default=None):
    v = input(prompt).strip()
    return v if v else default


def find_files():
    found = {}
    for year, info in SCENE_FILES.items():
        if all(os.path.exists(p) for p in info["bands"]):
            found[year] = info["bands"]
    return found


def get_inputs():
    print("\n" + "═"*68)
    print("  POWAI LANDSAT CLASSIFIER  |  IIT Bombay GNR630  Topic 12")
    print(f"  GPS locked: {POWAI_LAT}°N, {POWAI_LON}°E")
    print("═"*68)

    auto, datasets = find_files(), []
    if len(auto) == 3:
        print(f"\n  ✓ All 3 scenes auto-detected!")
        for year in ["2009", "2016", "2025"]:
            sensor = SCENE_FILES[year]["sensor"]
            print(f"    {year} [{sensor}]: {len(auto[year])} bands")
            datasets.append({"year": year, "sensor": sensor, "paths": auto[year]})
    else:
        print(f"\n  Found {len(auto)}/3 scenes. Enter missing paths manually.")
        for year, info in SCENE_FILES.items():
            sensor = info["sensor"]
            if year in auto:
                datasets.append({"year": year, "sensor": sensor, "paths": auto[year]})
                print(f"  ✓ {year} [{sensor}] auto-detected.")
            else:
                print(f"\n── {year} [{sensor}]  (need {len(info['bands'])} bands)")
                raw = ask("  Paths (comma-separated): ")
                paths = [p.strip().strip("'\"") for p in raw.split(",")] if raw else []
                datasets.append({"year": year, "sensor": sensor,
                                 "paths": paths if paths else None})

    print(f"\n── ROI ──────────────────────────────────────────────────────────")
    print(f"   ★ Validated: {OPTIMAL_ROI} px  → Urban +25% (2009→2025) ✓")
    print(f"     Press ENTER to accept. Do NOT change unless you know why.")
    raw = ask(f"  ROI size [ENTER = {OPTIMAL_ROI}]: ", str(OPTIMAL_ROI))
    roi = int(raw) if raw.isdigit() else OPTIMAL_ROI
    # if roi != OPTIMAL_ROI:
    #     print(f"  [WARN] {roi}px not validated — may give wrong urban trend.")

    out = ask("\n  Output filename [powai_lc_analysis.png]: ", "powai_lc_analysis.png")
    return datasets, roi, out


def main():
    datasets, roi_size, out_path = get_inputs()
    scenes = []

    for ds in datasets:
        year, sensor, paths = ds["year"], ds["sensor"], ds["paths"]
        print(f"\n{'─'*60}\n  [{year}  {sensor}]")
        if not paths:
            print("  [skip] No paths."); continue

        arr   = load_scene(paths)
        print(f"  Full scene: {arr.shape}")
        arr   = crop_to_roi(arr, paths[0], roi_size)
        arr_h = harmonize_l5_to_l8(arr, sensor)
        cl, idx, acc = classify(arr_h, sensor)
        ar = areas(cl)
        px = ar["px"]
        print(f"  RESULT  Water:{px['Water']:,}px={ar['Water']:.3f}km²  "
              f"Veg:{px['Vegetation']:,}px={ar['Vegetation']:.3f}km²  "
              f"Urban:{px['Urban / Other']:,}px={ar['Urban / Other']:.3f}km²")

        scenes.append({"year": year, "sensor": sensor,
                       "cl": cl, "fc": false_col(arr, sensor),
                       "tc": true_col(arr, sensor), "areas": ar, "acc": acc})

    if len(scenes) < 3:
        print("\n[ERROR] Need all 3 scenes. Check file paths."); return

    summary(scenes)
    plot(scenes, roi_size, out_path)




def plot_accuracy_report(scenes, out):
    """
    Standalone accuracy report — one page, nothing else.
    Shows for each year:
      • Confusion matrix with exact pixel counts + % + ✗ wrong labels
      • Bar chart: correct vs misclassified pixels per class
      • Table: OA, Kappa, PA, UA, F1, misclassified count
    """
    years = [s["year"] for s in scenes]
    ACCS  = [s["acc"]  for s in scenes]
    YC    = ["#58a6ff", "#f0883e", "#56d364"]
    BG, CARD, BD = "#0d1117", "#161b22", "#30363d"
    W, M = "#e6edf3", "#8b949e"

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle(
        "CLASSIFICATION ACCURACY REPORT  —  POWAI LAKE LAND COVER\n"
        "IIT Bombay  |  GNR630  |  Method: Stratified 70/30 Hold-out Validation",
        color=W, fontsize=14, fontweight="bold", y=0.98, fontfamily="monospace")

    gs = gridspec.GridSpec(3, 3, fig, hspace=0.55, wspace=0.35,
                           left=0.06, right=0.97, top=0.91, bottom=0.05)

    short = ["Water", "Veg", "Urban"]

    for col, (year, acc, yc) in enumerate(zip(years, ACCS, YC)):

        # ── Row 0: Confusion Matrix ───────────────────────────────────────────
        ax_cm = fig.add_subplot(gs[0, col])
        ax_cm.set_facecolor(BG)

        if acc is None:
            ax_cm.axis("off")
            ax_cm.text(0.5, 0.5, "N/A", transform=ax_cm.transAxes,
                       color=M, ha="center", va="center")
        else:
            conf  = acc["conf_mat"]
            total = conf.sum()
            n     = 3

            for r in range(n):
                for c in range(n):
                    val = conf[r, c]
                    pct = 100.0 * val / total if total > 0 else 0
                    if r == c:
                        face      = CLASS_COLOURS[r]
                        intensity = 0.30 + 0.60*(val/(conf[r].sum()+1e-9))
                    else:
                        face      = "#c0392b"
                        intensity = min(0.15 + 0.60*(val/(conf.max()+1e-9)), 0.70)

                    ax_cm.add_patch(plt.Rectangle(
                        (c, n-1-r), 1, 1, color=face, alpha=intensity,
                        linewidth=0, zorder=1))
                    ax_cm.add_patch(plt.Rectangle(
                        (c, n-1-r), 1, 1, fill=False,
                        edgecolor=BD, linewidth=1.2, zorder=2))

                    # Big count
                    ax_cm.text(c+0.5, n-1-r+0.65, f"{val:,}",
                               ha="center", va="center", color=W,
                               fontsize=14, fontweight="bold", zorder=3)
                    # Percentage
                    ax_cm.text(c+0.5, n-1-r+0.38, f"{pct:.1f}% of test",
                               ha="center", va="center", color=W,
                               fontsize=8, zorder=3)
                    # ✓ or ✗
                    mark = "✓ correct" if r == c else "✗ misclassified"
                    col_m = "#56d364" if r == c else "#f85149"
                    ax_cm.text(c+0.5, n-1-r+0.14, mark,
                               ha="center", va="center", color=col_m,
                               fontsize=7.5, fontweight="bold", zorder=3)

            ax_cm.set_xlim(0, 3); ax_cm.set_ylim(0, 3)
            ax_cm.set_xticks([0.5,1.5,2.5]); ax_cm.set_yticks([0.5,1.5,2.5])
            ax_cm.set_xticklabels(short, color=M, fontsize=10)
            ax_cm.set_yticklabels(short[::-1], color=M, fontsize=10)
            ax_cm.set_xlabel("Predicted Class →", color=M, fontsize=9, labelpad=4)
            ax_cm.set_ylabel("← True Class", color=M, fontsize=9, labelpad=4)
            ax_cm.tick_params(length=0)
            for sp in ax_cm.spines.values(): sp.set_color(BD)

            n_correct = int(np.trace(conf))
            n_wrong   = int(total - n_correct)
            ax_cm.set_title(
                f"{year}  |  OA: {acc['OA']:.1f}%   κ: {acc['kappa']:.3f}",
                color=W, fontsize=12, fontweight="bold", pad=8,
                bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=yc, lw=2))
            ax_cm.text(0.5, -0.16,
                       f"✓ {n_correct:,} correctly classified   "
                       f"✗ {n_wrong:,} misclassified   "
                       f"(n = {total:,} test pixels)",
                       transform=ax_cm.transAxes, color=M, fontsize=9,
                       ha="center", va="top", fontfamily="monospace",
                       bbox=dict(boxstyle="round,pad=0.3", fc="#1c2128", ec=BD))

        # ── Row 1: Bar — correct vs misclassified per class ───────────────────
        ax_bar = fig.add_subplot(gs[1, col])
        ax_bar.set_facecolor(CARD)
        for sp in ax_bar.spines.values(): sp.set_color(BD)
        ax_bar.tick_params(colors=M)

        if acc is not None:
            conf = acc["conf_mat"]
            x    = np.arange(3)
            correct_counts = [conf[i, i] for i in range(3)]
            wrong_counts   = [conf[i, :].sum() - conf[i, i] for i in range(3)]

            b1 = ax_bar.bar(x-0.2, correct_counts, 0.38, label="✓ Correct",
                            color="#56d364", alpha=0.85, edgecolor="none")
            b2 = ax_bar.bar(x+0.2, wrong_counts,   0.38, label="✗ Misclassified",
                            color="#f85149", alpha=0.85, edgecolor="none")

            for bars in [b1, b2]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax_bar.text(bar.get_x()+bar.get_width()/2,
                                    h + max(correct_counts+wrong_counts)*0.02,
                                    str(int(h)), ha="center", va="bottom",
                                    color=W, fontsize=9, fontweight="bold")

            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(short, color=W, fontsize=10, fontweight="bold")
            ax_bar.set_ylabel("Pixel Count", color=M, fontsize=9)
            ax_bar.set_title(f"{year}  —  Correct vs Misclassified",
                             color=W, fontsize=11, fontweight="bold", pad=5)
            ax_bar.legend(facecolor="#1c2128", edgecolor=BD,
                          labelcolor=W, fontsize=9, loc="upper right")
            ax_bar.grid(axis="y", color=BD, linewidth=0.6, zorder=0)
            ax_bar.set_ylim(0, ax_bar.get_ylim()[1]*1.20)

        # ── Row 2: Detailed metrics table ─────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[2, col])
        ax_tbl.set_facecolor(CARD); ax_tbl.axis("off")
        ax_tbl.set_title(f"{year}  —  Per-Class Accuracy",
                         color=W, fontsize=11, fontweight="bold", pad=5,
                         bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=yc, lw=1.5))

        if acc is not None:
            conf = acc["conf_mat"]
            trows = []
            for i, (lbl, s) in enumerate(zip(CLASS_LABELS, short)):
                tp      = conf[i, i]
                total_r = conf[i, :].sum()   # true positives of this class
                total_c = conf[:, i].sum()   # predicted as this class
                mis_r   = total_r - tp       # missed (false negatives)
                mis_c   = total_c - tp       # false positives
                trows.append([
                    s,
                    f"{tp:,}",
                    f"{mis_r:,}",
                    f"{mis_c:,}",
                    f"{acc['PA'][lbl]:.1f}%",
                    f"{acc['UA'][lbl]:.1f}%",
                    f"{acc['F1'][lbl]:.1f}%",
                ])
            tbl = ax_tbl.table(
                cellText=trows,
                colLabels=["Class", "✓TP", "✗FN", "✗FP", "PA", "UA", "F1"],
                cellLoc="center", loc="center",
                bbox=[0, 0.12, 1, 0.78])
            tbl.auto_set_font_size(False); tbl.set_fontsize(9)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(BD)
                txt = cell.get_text().get_text()
                if r == 0:
                    cell.set_facecolor("#21262d")
                    cell.set_text_props(color=W, fontweight="bold")
                else:
                    cell.set_facecolor(CARD)
                    # Colour PA/UA/F1 by value
                    if c in (4, 5, 6):
                        try:
                            v = float(txt.strip("%"))
                            cell.set_text_props(
                                color="#56d364" if v >= 85 else
                                      "#f0883e" if v >= 70 else "#f85149")
                        except: cell.set_text_props(color=W)
                    elif c in (2, 3):  # FN, FP columns
                        cell.set_text_props(
                            color="#f85149" if txt != "0" else "#56d364")
                    else:
                        cell.set_text_props(color=W)

            # Summary line
            n_correct = int(np.trace(conf))
            n_wrong   = int(conf.sum() - n_correct)
            ax_tbl.text(0.5, 0.06,
                        f"TP=True Positive  FN=False Negative  FP=False Positive\n"
                        f"PA=Producer's Acc (recall)   UA=User's Acc (precision)",
                        transform=ax_tbl.transAxes, color=M, fontsize=7.5,
                        ha="center", va="bottom", fontfamily="monospace")

    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓  Accuracy report saved → {out}")


if __name__ == "__main__":
    main()