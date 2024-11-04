# BEV_IFEM_ABM
This repository contains the basic data files and code for the BEV adoption ABM underpinning the dissertation Simulating Electric Vehicle Adoption Using Agent-Based Modeling

## Requirements

- Python 3.10.2
- MESA 2.1.2
- Pandas 1.4.1

## Data and Model Files Structure
The folder 'files' contains the necessary data files. Within 'files' is a folder 'model' containg the python files. Explainations of these items are:

 * [files](./files)
   * [x_df.pkl](./files/x_df.pkl) .pkl file of a pandas dataframe with the quantified IFEM scores
   * [time_series_data.pkl](./files/time_series_data.pkl) .pkl file of the time-series inputs to the model (prices, news, and target registration percents)
   * [fpop_final.pkl](./files/fpop_final.pkl) The full synthetic population for loading into the model
   * [fpop_connections_dict.pkl](./files/fpop_connections_dict.pkl) Forward connection dictionary for social networks among population
   * [fpop_reverse_connections_dict.pkl](./files/fpop_reverse_connections_dict.pkl) Reverse connection dictionary for social networks among population
   * [model](./files/model)
     * [run.py](./files/model/run.py) Initiates GUI of model
     * [server.py](./files/model/server.py) Supports GUI initiation of model
     * [model.py](./files/model/model.py) Contains the model. Run from terminal for batch runs and improved computational efficiency
     * [model-precision.ipynb](./files/model/model-precision.ipynb) Jupyter notebook for analyzing model BEV adoption against historical
    
Note 1: The fpop_final.pkl, fpop_connections_dict.pkl, and fpop_reverse_connections_dict.pkl files are too large for GitHub. The ones uploaded here are smaller example files. Contact me for access to the full files.

Note 2: MESA 2.1.2 has a known bug with Random.Activation that was discovered late in model construction. Use of python's random module avoids this problem in the model.

