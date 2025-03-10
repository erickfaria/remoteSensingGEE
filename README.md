# Remote Sensing with Google Earth Engine (GEE) 🌍

A Python repository for geospatial analysis, environmental monitoring, and remote sensing using Google Earth Engine.

## Overview

This repository provides tools, scripts, and workflows to leverage Google Earth Engine (GEE) for geospatial analysis and environmental monitoring. It includes pre-built functions, case studies, and Jupyter Notebook tutorials to help users process satellite imagery, calculate remote sensing indices, and analyze large-scale geospatial datasets.

## Getting Started

### Prerequisites

#### Google Earth Engine Account
To use this repository, you must have an approved Google Earth Engine (GEE) account.

1. [Sign up for GEE here](https://signup.earthengine.google.com/).
2. Once approved, log in to the [GEE Code Editor](https://code.earthengine.google.com/) to explore the JavaScript API.

#### Python Dependencies
Install required libraries:

```bash
pip install earthengine-api geopandas pandas geemap tqdm
```

### Repository Structure

```
remoteSensingGEE/
├── indexes/                  # Remote sensing indices (NDVI, SAVI, EVI, LAI, WSI, etc.)
│   ├── sentinel2IndexCalculator.py
│   └── ...
├── outils/                  # Authentication and utility scripts
│   ├── gee_auth.py          # Login and GEE initialization
│   └── .env.example         # Template for environment variables
├── case_studies/            # Practical examples (e.g., deforestation, crop monitoring)
├── examples/                # Example scripts showing how to use the tools
├── notebooks/               # Jupyter Notebook tutorials
├── config/                  # Configuration files (e.g., paths, regions)
└── README.md
```

## Features

### Pre-Built Functions:
* Calculate vegetation indices (NDVI, EVI, SAVI, LAI, WSI, etc.)
* Mask clouds and process Sentinel-2, Landsat, or MODIS imagery
* Batch processing and time-series analysis

### Case Studies:
* Land cover classification
* Drought and vegetation health monitoring
* Urban sprawl analysis

### Jupyter Notebooks:
* Step-by-step tutorials for GEE workflows
* Integration with Python libraries (Pandas, Geopandas, Matplotlib)

## Configuration

### Environment Setup

#### Authentication:
* The `outils/gee_auth.py` script handles GEE authentication.
* Do not commit your credentials! Instead:
  1. Copy `.env.example` to `.env` in the root directory.
  2. Add your GEE service account credentials to `.env`:
  ```
  SERVICE_ACCOUNT=your_service_account_email
  KEY_PATH=/path/to/your/service_account_key.json
  ```
  3. Add `.env` to `.gitignore`.

#### Customize Paths:
* Update `config/path.txt` with local paths to your data and credentials.

## Usage Example

```python
from outils.gee_auth import initialize_earth_engine
from indexes.sentinel2IndexCalculator import Sentinel2IndexCalculator
import ee

# Initialize GEE
initialize_earth_engine()

# Define region and dates
roi = ee.Geometry.Polygon([[[-57.0, -15.0], [-56.5, -15.0], [-56.5, -15.5], [-57.0, -15.5]]])
start_date = '01-01-2023'
end_date = '31-12-2023'

# Calculate indices
calculator = Sentinel2IndexCalculator(start_date, end_date, roi)
results_df = calculator.aggregate_all_batches()
print(results_df.head())
```

## Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-index`.
3. Commit changes: `git commit -m 'Add new vegetation index'`.
4. Push to the branch: `git push origin feature/new-index`.
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

* Google Earth Engine for providing satellite data and processing tools.
* geemap and earthengine-api communities for Python integration support.

## Contact

For questions or collaborations, please open an issue in this repository.

[Acesse o Balaio Científico](https://www.balaiocientifico.com)