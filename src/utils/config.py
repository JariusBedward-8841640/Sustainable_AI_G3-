import os

# Base Project Path (calculated relative to this file)
# This assumes config.py is in src/utils/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
REPORTS_FILE = os.path.join(REPORTS_DIR, 'energy_estimates.csv')

# --- Constants for Estimation ---
# Global Average Carbon Intensity (kg CO2 per kWh)
CARBON_INTENSITY_FACTOR = 0.475 

# Power Usage Effectiveness (PUE) standard for data centers
PUE = 1.67