import functools
import logging
import os
from pathlib import Path

from . import prompts
from . import tools

# Automatically load .env if available
try:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    if local_env.exists():
        load_dotenv(dotenv_path=local_env)
    load_dotenv()  # Also load project-root .env
except ImportError:
    pass

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")


@functools.cache
def _initialize_earth_engine():
    """Initializes the Earth Engine client exactly once."""
    try:
        import ee
        import google.auth

        if not _PROJECT_ID:
            logging.info("GOOGLE_CLOUD_PROJECT not set. Earth Engine will use fallback geospatial analysis.")
            return

        scopes = [
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
        credentials, _ = google.auth.default(scopes=scopes)

        ee.Initialize(
            credentials,
            project=_PROJECT_ID,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
        logging.info("Earth Engine initialized successfully for project: %s", _PROJECT_ID)

    except Exception as e:
        logging.warning("Earth Engine initialization skipped/failed: %s", e)


@functools.cache
def _initialize_model_backend():
    """Initializes LLM backend: prefers GEMINI_API_KEY, falls back to Vertex AI."""
    if _GEMINI_API_KEY:
        logging.info("Gemini API Key detected. Using Google AI Studio backend.")
        os.environ["GEMINI_API_KEY"] = _GEMINI_API_KEY
        return

    if _PROJECT_ID:
        try:
            import vertexai

            logging.info("Initializing Vertex AI for project: %s", _PROJECT_ID)
            vertexai.init(project=_PROJECT_ID)
            logging.info("Vertex AI initialized successfully.")
        except Exception as e:
            logging.warning("Vertex AI initialization skipped/failed: %s", e)
    else:
        logging.warning("Neither GEMINI_API_KEY nor GOOGLE_CLOUD_PROJECT is configured.")


def create_agent():
    """Instantiate the root Google ADK agent."""
    _initialize_earth_engine()
    _initialize_model_backend()

    agent_tools = [
        tools.get_geometry_area,
        tools.get_topography_risk_stats,
        tools.get_vegetation_fuel_stats,
        tools.get_canopy_structure_stats,
        tools.get_historical_burn_stats,
        tools.get_fire_danger_indices,
        tools.get_weathernext_fire_weather,
        tools.calculate_surface_fire_behavior,
    ]

    try:
        from google.adk.agents import llm_agent

        return llm_agent.Agent(
            name="wildfire_risk_agent",
            model="gemini-flash-latest",
            description="Agent to assess wildland fire risk, topography, fuels, weather, and spread dynamics from GeoJSON.",
            tools=agent_tools,
            instruction=prompts.root_prompt,
        )
    except ImportError:
        # Graceful fallback object if google-adk is not yet installed in local environment
        class LocalAgent:
            def __init__(self, name, model, description, tools, instruction):
                self.name = name
                self.model = model
                self.description = description
                self.tools = tools
                self.instruction = instruction

        return LocalAgent(
            name="wildfire_risk_agent",
            model="gemini-flash-latest",
            description="Agent to assess wildland fire risk, topography, fuels, weather, and spread dynamics from GeoJSON.",
            tools=agent_tools,
            instruction=prompts.root_prompt,
        )


root_agent = create_agent()
