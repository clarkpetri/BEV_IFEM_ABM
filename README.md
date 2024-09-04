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
   * [fpop_connections_dict.pkl](./files/fpop_connections_dict.pkl) For connection dictionary for social networks among population
   * [fpop_reverse_connections_dict.pkl](./files/fpop_reverse_connections_dict.pkl)
 * [dir1](./dir1)
   * [file11.ext](./dir1/file11.ext)
   * [file12.ext](./dir1/file12.ext)
 * [file_in_root.ext](./file_in_root.ext)
 * [README.md](./README.md)
 * [dir3](./dir3)
