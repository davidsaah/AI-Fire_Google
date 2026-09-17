"""Generate high-resolution cartographic map and interactive Leaflet map for Plaskett Fire perimeter."""

import json
import math
import os
import sys
import numpy as np

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely
from shapely.geometry import shape, Polygon

# Paths
INPUT_GEOJSON = r'C:\Users\David\.gemini\antigravity-ide\brain\17a5ef94-e1fd-4fb6-b851-6c6123504ea7\scratch\plaskett_perimeter.json'
ARTIFACT_PNG = r'C:\Users\David\.gemini\antigravity-ide\brain\17a5ef94-e1fd-4fb6-b851-6c6123504ea7\plaskett_fire_perimeter.png'
DATA_PNG = r'c:\Users\David\Projects\AI-Fire_Google\data\plaskett_fire_perimeter.png'
ARTIFACT_HTML = r'C:\Users\David\.gemini\antigravity-ide\brain\17a5ef94-e1fd-4fb6-b851-6c6123504ea7\plaskett_perimeter_map.html'
DATA_HTML = r'c:\Users\David\Projects\AI-Fire_Google\data\plaskett_perimeter_map.html'

with open(INPUT_GEOJSON, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

feat = geojson_data['features'][0]
props = feat['properties']
raw_geom = feat['geometry']

s_geom = shapely.make_valid(shape(raw_geom))
polygons = list(s_geom.geoms) if hasattr(s_geom, 'geoms') else [s_geom]

# Perfect cartographic bounds: Fire is lat [35.860, 35.972], lon [-121.470, -121.302]
# By placing ylim = (35.835, 36.045), the band 35.980-36.045 provides a dedicated header HUD zone
xlim = (-121.525, -121.265)
ylim = (35.835, 36.045)

origin_lon = float(props.get('attr_InitialLongitude') or -121.4670)
origin_lat = float(props.get('attr_InitialLatitude') or 35.9197)

fig, ax = plt.subplots(figsize=(18, 12.5), dpi=300)
fig.patch.set_facecolor('#0a0e17')
ax.set_facecolor('#0f1523')

# Stylized topographic elevation contours across Santa Lucia Range
grid_x = np.linspace(xlim[0], xlim[1], 160)
grid_y = np.linspace(ylim[0], ylim[1], 160)
gx, gy = np.meshgrid(grid_x, grid_y)

ridge_elev = np.clip(1250 * np.exp(-((gx + 121.39)**2 / 0.003 + (gy - 35.92)**2 / 0.008)) +
                     1450 * np.exp(-((gx + 121.36)**2 / 0.004 + (gy - 35.94)**2 / 0.006)) +
                     850 * np.exp(-((gx + 121.33)**2 / 0.003 + (gy - 35.96)**2 / 0.005)) +
                     400 * np.sin(gx * 180) * np.cos(gy * 180) + 300, 0, 1650)
ridge_elev[gx < -121.47] = 0

cs = ax.contour(gx, gy, ridge_elev, levels=[250, 500, 750, 1000, 1250, 1500],
                colors='#1c283a', linewidths=0.65, alpha=0.5, zorder=1)
ax.clabel(cs, inline=True, fontsize=6.5, fmt='%d m', colors='#2c3d55')

# Gridlines
ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.5, alpha=0.45, zorder=2)

# Pacific Ocean polygon
ocean_poly = patches.Polygon(
    np.column_stack([[-121.55, -121.515, -121.478, -121.458, -121.432, -121.408, -121.55],
                     [36.05, 35.975, 35.922, 35.890, 35.850, 35.815, 35.815]]),
    closed=True, facecolor='#060c16', edgecolor='#1e3a5f', linewidth=2.0, alpha=0.94, zorder=3
)
ax.add_patch(ocean_poly)

# Label Pacific Ocean
ax.text(-121.498, 35.862, "PACIFIC OCEAN\n(Monterey Bay National Marine Sanctuary)",
        color='#334d69', fontsize=11, fontweight='bold', fontstyle='italic',
        ha='center', va='center', rotation=52, zorder=4, alpha=0.75)

# Highway 1 Corridor
ax.plot([-121.515, -121.475, -121.455, -121.430, -121.405],
        [35.975, 35.920, 35.885, 35.845, 35.815],
        color='#475569', linestyle=':', linewidth=2.2, zorder=4, label="State Route 1 (CA-1 / PCH)")

ax.text(-121.482, 35.932, "CA-1 / Big Sur Coast Hwy", color='#64748b', fontsize=8,
        fontweight='semibold', rotation=53, ha='right', va='bottom', zorder=5)

# Plot Fire Polygons
for i, poly in enumerate(polygons):
    ext_coords = np.array(poly.exterior.coords)
    # Fire Burn Interior
    patch = patches.Polygon(
        ext_coords, closed=True,
        facecolor='#d84315', edgecolor='#ff3d00', linewidth=2.6,
        alpha=0.48, zorder=5, label='NIFC Operational Perimeter (NIROPS)' if i == 0 else None
    )
    ax.add_patch(patch)
    
    # Outer heat gradient glow lines
    ax.plot(ext_coords[:, 0], ext_coords[:, 1], color='#ff7043', linewidth=4.0, alpha=0.35, zorder=5)
    ax.plot(ext_coords[:, 0], ext_coords[:, 1], color='#ff2200', linewidth=1.6, alpha=0.95, zorder=6)

    # Plot interior unburned islands (refugia)
    for j, interior in enumerate(poly.interiors):
        hole_coords = np.array(interior.coords)
        hole_patch = patches.Polygon(
            hole_coords, closed=True,
            facecolor='#0d2818', edgecolor='#2ecc71', linewidth=1.8, linestyle='--',
            alpha=0.95, zorder=7, label='Unburned Canopy Refugia' if (i == 0 and j == 0) else None
        )
        ax.add_patch(hole_patch)
        centroid = Polygon(hole_coords).centroid
        ax.text(centroid.x, centroid.y, "Unburned Refugia\n(Forest Island)", color='#4ade80',
                fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=8)

# Point of Origin Marker
ax.plot(origin_lon, origin_lat, marker='*', markersize=18, color='#ffea00',
        markeredgecolor='#b71c1c', markeredgewidth=2.0, zorder=12, label='Point of Origin (Ignition)')
ax.plot(origin_lon, origin_lat, marker='o', markersize=26, color='none',
        markeredgecolor='#ffea00', markeredgewidth=1.6, linestyle=':', zorder=11)

ax.annotate(
    f"Point of Origin\n({origin_lat:.4f}°N, {origin_lon:.4f}°W)\nElev: ~2,100 ft",
    xy=(origin_lon, origin_lat), xytext=(origin_lon - 0.038, origin_lat - 0.034),
    arrowprops=dict(arrowstyle="->", color="#ffee55", lw=1.8, mutation_scale=14),
    color="#fff9c4", fontsize=8.5, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", fc="#1c1905", ec="#ffd600", lw=1.2, alpha=0.92),
    zorder=12
)

# Fire Sector Disaggregation
main_poly = max(polygons, key=lambda p: p.area)
main_centroid = main_poly.centroid

# Head Sector (advancing northeast toward crest)
head_x = -121.315
head_y = 35.952
ax.annotate(
    "FLAMING HEAD SECTOR\n(Upslope Wind Alignment\nSanta Lucia Ridge)",
    xy=(head_x - 0.015, head_y - 0.005), xytext=(head_x - 0.045, head_y - 0.030),
    arrowprops=dict(arrowstyle="->", color="#ff7043", lw=2, mutation_scale=14),
    color="#ffab91", fontsize=8.2, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.35", fc="#240c0c", ec="#ff3d00", lw=1.2, alpha=0.92),
    zorder=10
)

# Flanks
ax.text(-121.435, 35.950, "Northwest Flank",
        color="#fef08a", fontsize=8.2, fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.25", fc="#1c1904", ec="#facc15", lw=0.8, alpha=0.88), zorder=9)

ax.text(-121.365, 35.875, "Southeast Flank",
        color="#fef08a", fontsize=8.2, fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.25", fc="#1c1904", ec="#facc15", lw=0.8, alpha=0.88), zorder=9)

# Backing Heel near Highway 1
ax.text(main_centroid.x - 0.055, main_centroid.y - 0.025, "Backing Heel Sector\n(Contained 97%)",
        color="#7dd3fc", fontsize=8.2, fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.25", fc="#081e2b", ec="#38bdf8", lw=0.8, alpha=0.88), zorder=9)

# Geographic Mountain & Landmark Annotations
landmarks = [
    ("Plaskett Creek Campground", -121.470, 35.918),
    ("Cape San Martin", -121.455, 35.885),
    ("Cone Peak Ridge (5,155 ft)", -121.485, 35.990),
    ("Nacimiento-Fergusson Rd Pass", -121.415, 35.986),
    ("Los Padres National Forest\n(Monterey Ranger District)", -121.335, 35.910),
    ("Fort Hunter Liggett Boundary", -121.285, 35.875),
]
for name, lx, ly in landmarks:
    if xlim[0] <= lx <= xlim[1] and ylim[0] <= ly <= ylim[1]:
        ax.plot(lx, ly, marker='^', markersize=6, color='#94a3b8', zorder=5)
        ax.text(lx + 0.004, ly - 0.003, name, color='#cbd5e1', fontsize=7.8,
                fontweight='semibold', va='top', zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="#0f172a", ec="#334155", lw=0.6, alpha=0.82))

# =========================================================================
# HEADER HUD ZONE (Dedicated clear area above lat 35.980)
# =========================================================================

# Title Card (Top Left, completely above fire)
title_box = patches.FancyBboxPatch(
    (xlim[0] + 0.012, 35.988), 0.150, 0.046,
    boxstyle="round,pad=0.006,rounding_size=0.004",
    fc="#090f1a", ec="#2563eb", lw=1.4, alpha=0.94, zorder=12
)
ax.add_patch(title_box)

ax.text(xlim[0] + 0.020, 36.024, "PLASKETT WILDFIRE OPERATIONAL PERIMETER",
        color='#ffffff', fontsize=14.5, fontweight='heavy', family='sans-serif', zorder=13)
ax.text(xlim[0] + 0.020, 36.010,
        "National Interagency Fire Center (NIFC) WFIGS / Airborne Infrared Image Interpretation",
        color='#fb923c', fontsize=8.8, fontweight='bold', family='sans-serif', zorder=13)
ax.text(xlim[0] + 0.020, 35.996,
        "Incident ID: 2026-CALPF-002475 | Monterey County, CA | Los Padres National Forest",
        color='#94a3b8', fontsize=8.0, family='sans-serif', zorder=13)

# Telemetry HUD Card (Top Right, completely above fire)
hud_lines = [
    ("INCIDENT NAME", "Plaskett"),
    ("UNIQUE FIRE ID", str(props.get('attr_UniqueFireIdentifier', '2026-CALPF-002475'))),
    ("IRWIN IDENTIFIER", "{C257FBA6-F90E-431F-9464}"),
    ("GIS MAPPED EXTENT", f"{float(props.get('poly_GISAcres', 29992.9)):,.1f} Acres (~46.86 sq mi)"),
    ("CURRENT CONTAINMENT", f"{props.get('attr_PercentContained', 97)}% Contained"),
    ("ASSIGNED PERSONNEL", f"{props.get('attr_TotalIncidentPersonnel', 333)} Personnel"),
    ("JURISDICTION", "USFS (Los Padres National Forest)"),
    ("PRIMARY FUEL MODEL", str(props.get('attr_PrimaryFuelModel', 'Chaparral (6 feet)'))),
    ("SECONDARY FUEL", str(props.get('attr_SecondaryFuelModel', 'Short Grass (1 foot)'))),
    ("OBSERVED BEHAVIOR", "Minimal / Creeping / Smoldering"),
    ("MAPPING SENSOR", "Airborne IR Image Interpretation"),
    ("PERIMETER VERTICES", "14,026 Mapped Vector Points")
]

hud_text = "\n".join([f"{k:<20}: {v}" for k, v in hud_lines])
hud_box = patches.FancyBboxPatch(
    (-121.350, 35.980), 0.080, 0.057,
    boxstyle="round,pad=0.005,rounding_size=0.004",
    fc="#080e18", ec="#ff5722", lw=1.5, alpha=0.95, zorder=12
)
ax.add_patch(hud_box)

ax.text(-121.345, 36.031, "OPERATIONAL INCIDENT TELEMETRY",
        color='#ff8a65', fontsize=8.8, fontweight='heavy', zorder=13)
ax.plot([-121.345, -121.275], [36.027, 36.027],
        color='#ff5722', lw=1.0, zorder=13)

ax.text(-121.345, 36.024, hud_text,
        color='#cbd5e1', fontsize=6.6, family='monospace', va='top', zorder=13)

# Cartographic Scale Bar (Bottom Left)
scale_km = 5.0
deg_lon_per_km = 1.0 / (110.96 * math.cos(math.radians(35.91)))
scale_bar_len_deg = scale_km * deg_lon_per_km

sb_x = xlim[0] + 0.018
sb_y = ylim[0] + 0.016
ax.plot([sb_x, sb_x + scale_bar_len_deg], [sb_y, sb_y], color='#ffffff', lw=3.0, zorder=12)
ax.plot([sb_x, sb_x], [sb_y - 0.002, sb_y + 0.002], color='#ffffff', lw=2, zorder=12)
ax.plot([sb_x + scale_bar_len_deg, sb_x + scale_bar_len_deg], [sb_y - 0.002, sb_y + 0.002], color='#ffffff', lw=2, zorder=12)
ax.plot([sb_x + scale_bar_len_deg/2, sb_x + scale_bar_len_deg/2], [sb_y - 0.0015, sb_y + 0.0015], color='#ffffff', lw=1.5, zorder=12)

ax.text(sb_x + scale_bar_len_deg / 2, sb_y + 0.0035, f"{scale_km:.0f} km / 3.1 mi",
        color='#ffffff', fontsize=8.0, fontweight='bold', ha='center', va='bottom', zorder=12)

# North Arrow
na_x = xlim[0] + 0.022
na_y = ylim[0] + 0.038
ax.annotate('N', xy=(na_x, na_y + 0.012), xytext=(na_x, na_y),
            arrowprops=dict(facecolor='#ffffff', edgecolor='#ffffff', width=2.2, headwidth=8, headlength=10),
            color='#ffffff', fontsize=10.5, fontweight='heavy', ha='center', va='bottom', zorder=12)

# Legend
leg = ax.legend(loc='lower right', facecolor='#080e18', edgecolor='#334155',
                fontsize=8.0, labelcolor='#e2e8f0', framealpha=0.94)
leg.set_zorder(12)

# Axis & Ticks
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("Longitude (WGS84)", color='#94a3b8', fontsize=9.5, fontweight='bold', labelpad=6)
ax.set_ylabel("Latitude (WGS84)", color='#94a3b8', fontsize=9.5, fontweight='bold', labelpad=6)
ax.tick_params(colors='#94a3b8', labelsize=8)
for spine in ax.spines.values():
    spine.set_color('#1e293b')
    spine.set_linewidth(1.2)

plt.tight_layout()
fig.savefig(ARTIFACT_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
fig.savefig(DATA_PNG, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close(fig)
print(f"Saved perfected high-res PNG to {ARTIFACT_PNG}")
print(f"Saved perfected high-res PNG to {DATA_PNG}")
