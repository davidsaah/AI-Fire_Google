# Wildfire Risk Assessment & Spread Prediction Agent 🔥

An operational wildland fire behavior and geospatial risk assessment agent powered by **Google ADK** (`google-adk`), **Google Earth Engine** (`earthengine-api`), and **Google DeepMind WeatherNext 3**.

Modeled after the Google Earth Engine Community ADK Agent architecture, this agent allows users to input any **GeoJSON polygon** (e.g., a community, parcel, watershed, or national forest) and receive a comprehensive wildland fire hazard report.

---

## Capabilities & Analysis Suite

The agent evaluates:
1. **Scale & Topography**: Total acreage, elevation range, slope gradient (%), and dominant aspect.
2. **Fuel Bed & Canopy Structure**: Surface fuel classification (grass, shrub/chaparral, timber litter) and Canopy Bulk Density (CBD) for crown fire susceptibility.
3. **Historical Fire Regimes**: Burn scars and fire return intervals over the past 15-20 years.
4. **Fire Danger Indices**: Energy Release Component (ERC percentile) and 100-hour dead fuel moisture from GridMET.
5. **Real-Time Fire Weather**: 10m wind speed, gusts, direction, temperature, and RH from Google DeepMind's WeatherNext 3 model.
6. **Fire Spread Physics**: Computes forward Rate of Spread (m/min and ch/hr), Flame Length (m/ft), Fireline Intensity (kW/m), and spotting risk.
7. **Mitigation Action Plan**: Defensible space priority zones and suppression containment difficulty.

---

## Setup & Installation

### 1. Prerequisites
* Python 3.10+
* A Google Cloud Project with the **Earth Engine API** and **Vertex AI API** enabled.
* Google Cloud CLI (`gcloud`).

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install google-adk
```

### 3. Configuration & Authentication
Set your project in `wildfire_risk_agent/.env`:
```env
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
```

Authenticate with Google Cloud and Earth Engine:
```bash
gcloud auth application-default login
earthengine authenticate
```

---

## Running the Agent

You can interact with the agent either via CLI or in a browser interface:

### Option A: Command Line Interface (CLI)
```bash
adk run wildfire_risk_agent
```

### Option B: Web Chat Interface
```bash
adk web
```
Then select **`wildfire_risk_agent`** from the dropdown menu.

---

## Example Interaction

In the chat prompt, provide a GeoJSON polygon of your area of interest (e.g. an area in Napa County / Calistoga, CA):

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [-122.585, 38.565],
        [-122.555, 38.565],
        [-122.555, 38.595],
        [-122.585, 38.595],
        [-122.585, 38.565]
      ]
    ]
  },
  "properties": {
    "name": "Diamond Mountain Fire Risk Area"
  }
}
```

The agent will execute its 6-phase analytical protocol and respond with a structured **Wildfire Hazard & Risk Assessment Report**!
