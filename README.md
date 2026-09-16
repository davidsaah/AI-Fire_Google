# AI-Fire 🔥

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Simulation Core: Pyretechnics](https://img.shields.io/badge/core-Pyretechnics-orange.svg)](https://github.com/pyregence/pyretechnics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AI-Fire** is an artificial intelligence and machine learning framework built on top of the [Pyretechnics](https://github.com/pyregence/pyretechnics) wildland fire simulation engine. It bridges operational physics-based wildfire behavior models with modern deep learning, fast surrogate modeling, real-time satellite data assimilation, and intelligent wildfire risk forecasting.

---

## Table of Contents

- [Overview](#overview)
- [Why Pyretechnics?](#why-pyretechnics)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Project Directory Structure](#project-directory-structure)
- [Prerequisites & Installation](#prerequisites--installation)
- [Quickstart Guide](#quickstart-guide)
  - [1. Single-Cell Fire Behavior Calculation](#1-single-cell-fire-behavior-calculation)
  - [2. Multi-Dimensional Landscape Fire Spread (Level-Set)](#2-multi-dimensional-landscape-fire-spread-level-set)
  - [3. AI-Fire Ensemble & Surrogate Workflow](#3-ai-fire-ensemble--surrogate-workflow)
- [Data Ingestion & Pipeline](#data-ingestion--pipeline)
- [Roadmap](#roadmap)
- [References & Acknowledgments](#references--acknowledgments)
- [License](#license)

---

## Overview

Accurate and timely wildland fire modeling is vital for community protection, ecological management, and emergency response. However, traditional operational fire spread models can be computationally prohibitive when running high-resolution, Monte Carlo ensemble simulations across large geographic extents.

**AI-Fire** pairs the rigorous, validated physical equations implemented in **Pyretechnics** with modern AI techniques:
- **Physics-Informed Surrogates**: Accelerating fire perimeter propagation predictions by orders of magnitude via neural operators and convolutional architectures.
- **Ensemble Simulation Automation**: Effortlessly generating tens of thousands of simulated fire spread scenarios under stochastic weather and fuel conditions.
- **Real-Time Data Assimilation**: Ingesting thermal satellite detections (e.g., VIIRS, MODIS, GOES) and weather forecasts (e.g., HRRR, ERA5) to nudge and calibrate active fire fronts.
- **Decision Support & Optimization**: Evaluating fuel treatment efficacy, containment line placement, and evacuation planning with reinforcement learning.

---

## Why Pyretechnics?

[Pyretechnics](https://github.com/pyregence/pyretechnics) is an open-source, scriptable Python/Cython fire modeling engine developed by the [Pyregence Consortium](https://pyregence.org). It re-implements and unifies the core fire science algorithms from legacy modeling systems (such as FARSITE, FlamMap, BehavePlus, and ELMFIRE) into a clean, modern Python interface.

### Core Pyretechnics Modules

| Module | Description | Scientific Basis |
| :--- | :--- | :--- |
| `pyretechnics.fuel_models` | Fuel characteristics and moisture dynamics | Scott & Burgan (40 models), Anderson (13 models) |
| `pyretechnics.surface_fire` | Rate of spread, reaction intensity, flame length | Rothermel (1972), Albini (1976) |
| `pyretechnics.crown_fire` | Active/passive crown fire initiation and spread | Cruz, Alexander & Wakimoto (2005), Van Wagner (1977) |
| `pyretechnics.spot_fire` | Ember generation, trajectory, and spotting distance | Albini (1979) |
| `pyretechnics.burn_cells` | Vectorized cellular fire metrics across grids | Gridded cellular evaluation |
| `pyretechnics.eulerian_level_set` | Continuous fire front propagation and arrival times | ELMFIRE Eulerian Level-Set formulation |
| `pyretechnics.space_time_cube` | Harmonization of 0D–3D spatial and temporal data layers | Unified multi-resolution raster cube |

---

## System Architecture

```
                       +-----------------------------------+
                       |    Raw Geospatial Data Streams    |
                       |  LANDFIRE (Fuels) | HRRR (Weather)|
                       |  USGS 3DEP (DEM)  | VIIRS/GOES    |
                       +-----------------+-----------------+
                                         |
                                         v
                       +-----------------------------------+
                       |      Data Ingestion & Cube        |
                       |   `pyretechnics.space_time_cube`  |
                       +-----------------+-----------------+
                                         |
                +------------------------+------------------------+
                |                                                 |
                v                                                 v
+-------------------------------+               +----------------------------------+
|   Pyretechnics Physics Core   |               |     AI-Fire ML / DL Engine       |
|  - Surface / Crown Equations  |               |  - Neural Surrogate Models       |
|  - Spot Fire Mechanics        |  (Ensembles)  |  - Deep Learning Spread Forecast |
|  - Eulerian Level-Set Spread  | ------------> |  - Active Firefront Nudging      |
|  - Time of Arrival (TOA) Grids|               |  - Uncertainty Quantification    |
+-------------------------------+               +----------------------------------+
                |                                                 |
                +------------------------+------------------------+
                                         |
                                         v
                       +-----------------------------------+
                       |      Analysis & Visualization     |
                       |  GeoTIFF Outputs, Perimeter Maps, |
                       |  Burn Probabilities, GeoJSON API  |
                       +-----------------------------------+
```

---

## Key Features

- **Pure Python & NumPy Native**: Integrates seamlessly with the scientific Python ecosystem (`scipy`, `rasterio`, `geopandas`, `xarray`, `torch`, `jax`).
- **Flexible Spatiotemporal Input**: Handles static rasters, time-varying weather grids, and point weather observations uniformly via `SpaceTimeCube`.
- **Modular Physics**: Select between single-cell reaction calculations, gridded raster simulations, or level-set front tracking.
- **Machine Learning Ready**: Prepares training datasets (inputs: fuel, elevation, slope, aspect, wind U/V, fuel moisture; targets: arrival time, flame length, fireline intensity) directly as tensors.
- **Scalable Batch Processing**: Designed to distribute thousands of independent ignition runs across multi-core CPUs and GPU clusters.

---

## Project Directory Structure

```text
AI-Fire/
├── configs/                  # Simulation, model, and data configuration files (YAML/JSON)
│   ├── default_simulation.yaml
│   └── fuel_models.yaml
├── data/                     # Data cache directory (ignored by git)
│   ├── raw/                  # Downloaded LANDFIRE, DEM, weather rasters
│   ├── processed/            # Harmonized SpaceTimeCube inputs
│   └── scenarios/            # Pre-configured test fire scenarios
├── notebooks/                # Jupyter exploration and demonstration notebooks
│   ├── 01_pyretechnics_basics.ipynb
│   ├── 02_landscape_simulation.ipynb
│   └── 03_ai_surrogate_training.ipynb
├── src/
│   └── ai_fire/              # Core AI-Fire package
│       ├── __init__.py
│       ├── core/             # Pyretechnics wrappers and simulation runners
│       │   ├── __init__.py
│       │   ├── level_set.py
│       │   └── simulator.py
│       ├── data/             # Spatial data pipelines (LANDFIRE, HRRR, DEM)
│       │   ├── __init__.py
│       │   ├── fetcher.py
│       │   └── cube_builder.py
│       ├── models/           # AI / Machine Learning surrogate architectures
│       │   ├── __init__.py
│       │   ├── surrogate.py
│       │   └── assimilation.py
│       └── utils/            # Geometry, raster I/O, and plotting helpers
│           ├── __init__.py
│           ├── plotting.py
│           └── raster.py
├── tests/                    # Unit and integration test suite
│   ├── test_physics.py
│   └── test_simulator.py
├── .gitignore
├── environment.yml           # Conda environment definition
├── pyproject.toml            # Project build and dependency specifications
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Prerequisites & Installation

### 1. Prerequisites

- Python 3.10 or higher
- C/C++ compiler toolchain (required for building Cython extensions if building from source)
- GDAL / PROJ libraries (recommended for geospatial raster processing)

### 2. Setting Up Virtual Environment

Using `conda` / `mamba` (recommended for geospatial dependencies):

```bash
# Create and activate environment
conda create -n ai-fire python=3.11 -y
conda activate ai-fire

# Install geospatial dependencies
conda install -c conda-forge gdal rasterio geopandas -y
```

Or using standard `venv`:

```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Pyretechnics & AI-Fire Dependencies

Install `pyretechnics` from PyPI:

```bash
pip install pyretechnics
```

Install AI-Fire in editable development mode:

```bash
git clone https://github.com/<your-org-or-user>/AI-Fire.git
cd AI-Fire
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart Guide

### 1. Single-Cell Fire Behavior Calculation

Compute surface fire spread rate and flame length for a specific fuel type under given moisture, wind, and slope conditions:

```python
import numpy as np
import pyretechnics.fuel_models as fm
import pyretechnics.surface_fire as sf

# Select a Scott & Burgan fuel model (e.g., Short Grass, Moderate Load - GS2)
fuel_model = fm.get_fuel_model("GS2")

# Define environmental parameters
dead_fuel_moisture = np.array([0.06, 0.07, 0.08])  # 1-hr, 10-hr, 100-hr
live_fuel_moisture = np.array([0.60, 0.90])        # Live herb, live woody
midflame_wind_speed = 12.0                         # mph
slope_steepness = 0.15                             # rise / run (fraction)

# Compute surface fire behavior
behavior = sf.compute_surface_fire(
    fuel_model=fuel_model,
    dead_moisture=dead_fuel_moisture,
    live_moisture=live_fuel_moisture,
    wind_speed=midflame_wind_speed,
    slope=slope_steepness,
)

print(f"Spread Rate: {behavior['rate_of_spread']:.2f} m/min")
print(f"Flame Length: {behavior['flame_length']:.2f} m")
print(f"Fireline Intensity: {behavior['fireline_intensity']:.2f} kW/m")
```

---

### 2. Multi-Dimensional Landscape Fire Spread (Level-Set)

Simulate continuous perimeter expansion over a gridded landscape using the Eulerian Level-Set method:

```python
import numpy as np
from pyretechnics.space_time_cube import SpaceTimeCube
from pyretechnics.eulerian_level_set import simulate_fire_spread

# 1. Prepare landscape grids (dimensions: Y, X)
grid_shape = (200, 200)
cell_resolution = 30.0  # 30-meter resolution (LANDFIRE standard)

elevation = np.zeros(grid_shape, dtype=np.float32)       # Flat or DEM array
fuel_codes = np.full(grid_shape, 102, dtype=np.int32)    # Fuel model code (e.g. GS2)
wind_speed = np.full(grid_shape, 15.0, dtype=np.float32) # Wind speed (mph)
wind_dir = np.full(grid_shape, 270.0, dtype=np.float32)  # Wind coming from West (270°)

# 2. Package into a SpaceTimeCube
cube = SpaceTimeCube(
    elevation=elevation,
    fuel_models=fuel_codes,
    wind_speed=wind_speed,
    wind_direction=wind_dir,
    resolution=cell_resolution,
)

# 3. Define ignition location (X, Y in grid coordinates)
ignition_mask = np.zeros(grid_shape, dtype=bool)
ignition_mask[100, 100] = True

# 4. Execute fire spread simulation
simulation_duration_minutes = 180.0  # 3 hours
results = simulate_fire_spread(
    space_time_cube=cube,
    ignition=ignition_mask,
    duration=simulation_duration_minutes,
)

# Extract Time of Arrival (TOA) grid
time_of_arrival = results["time_of_arrival"]
print(f"Simulation completed. Total burned cells: {np.sum(time_of_arrival > 0)}")
```

---

### 3. AI-Fire Ensemble & Surrogate Workflow

Generate high-throughput simulation runs to train a fast convolutional or graph surrogate model:

```python
from ai_fire.core.simulator import FireEnsembleRunner
from ai_fire.models.surrogate import FireSpreadSurrogate

# Initialize ensemble generator with spatial domain and variable distributions
runner = FireEnsembleRunner(
    base_landscape="data/scenarios/demo_landscape.tif",
    wind_speed_range=(5.0, 30.0),
    moisture_range=(0.04, 0.16),
)

# Run parallel simulations with Pyretechnics
dataset = runner.generate_samples(num_simulations=1000, n_jobs=-1)

# Train neural surrogate emulator
surrogate = FireSpreadSurrogate()
surrogate.fit(dataset, epochs=50)

# Real-time inference (<10ms)
predicted_toa = surrogate.predict(new_conditions)
```

---

## Data Ingestion & Pipeline

AI-Fire is built to ingest standard public wildfire data layers:

| Layer | Source | Format | Purpose |
| :--- | :--- | :--- | :--- |
| **Fuel Models** | [LANDFIRE](https://www.landfire.gov/) FBFM40 / FBFM13 | GeoTIFF | Surface fuel loads, surface-area-to-volume ratio |
| **Topography** | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | GeoTIFF | Elevation, slope angle, and aspect calculations |
| **Canopy Characteristics** | [LANDFIRE](https://www.landfire.gov/) (CBD, CBH, CC, CH) | GeoTIFF | Crown fire initiation and active crowning thresholds |
| **Weather & Winds** | [NOAA HRRR / RTMA / ERA5](https://rapidrefresh.noaa.gov/hrrr/) | GRIB2 / NetCDF | Spatiotemporal wind vectors, temperature, RH |
| **Fuel Moisture** | National Fuel Moisture Database / GridMET | CSV / GeoTIFF | 1-hr, 10-hr, 100-hr dead and live fuel moistures |
| **Thermal Detections** | FIRMS (VIIRS / MODIS) | Shapefile / GeoJSON | Active fire ignition points, perimeter calibration |

---

## Roadmap

- [x] Integrate `pyretechnics` core physics equations and Level-Set algorithm.
- [ ] Implement automated LANDFIRE and NOAA HRRR automated raster fetchers.
- [ ] Build high-throughput Monte Carlo simulation orchestrator.
- [ ] Train physics-informed neural surrogate model (ConvLSTM / Fourier Neural Operator).
- [ ] Develop satellite thermal detection assimilation (VIIRS active perimeter nudge).
- [ ] Interactive Streamlit / Web-based visualization and scenario dashboard.

---

## References & Acknowledgments

- **Pyretechnics**: A Python library for simulating fire behavior developed by the [Pyregence Consortium](https://pyregence.org). Repository: [pyregence/pyretechnics](https://github.com/pyregence/pyretechnics).
- **Rothermel, R. C. (1972)**: *A mathematical model for predicting fire spread in wildland fuels*. USDA Forest Service Res. Pap. INT-115.
- **Cruz, M. G., Alexander, M. E., & Wakimoto, R. H. (2005)**: *Development and testing of models for predicting crown fire rate of spread in conifer forest stands*. International Journal of Wildland Fire.
- **ELMFIRE**: Eulerian Level-set Model for Fire spread by Lautenberger, C. (2013).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
