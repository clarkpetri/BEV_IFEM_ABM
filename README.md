# BEV_IFEM_ABM
This repository contains the basic data files and code for the BEV adoption ABM underpinning the dissertation Simulating Electric Vehicle Adoption Using Agent-Based Modeling

## Requirements

- Python 3.10.2
- MESA 2.1.2
- Pandas 1.4.1

## Data and Model Files Structure
├── data
│   ├── fpop_final.pkl
│   ├── time_series_data.pkl
│   └── other_data_files.pkl
└── Model
    ├── run.py        # For visualizing the model with MESA's GUI
    ├── server.py     # Supports the GUI visualization
    └── model.py      # Contains the core model code
