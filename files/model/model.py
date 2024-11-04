import mesa
import random
import pandas as pd
import pickle
import numpy as np
import time
import os
from datetime import datetime

sample = True
agent_data = False # Output agent BEV adoption data
start_tick = 65
now = datetime.now()
date_time = now.strftime("%b %d %H%M")
#dir_name = 'Output data/03 - Optimized Params 82x82 full pop 8it'
#dir_name = 'Output Data/01 - Baseline all 1.0 params 90k 8it/'
#dir_name = f'Output Data/{date_time} 30x30 3.9 param sweep 90k 1it/'
dir_name = 'Output data/04 - eh_var sensitivity'

# Intended Data Structure #
# model/
# ├── model vX.XX/
# │   └── model.py
# ├── x_df.pkl
# ├── time_series_data.pkl
# └── Output Data/NAME/

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Path to the parent directory containing the data files
read_data_directory = os.path.join(script_directory, '..')

write_data_directory = os.path.join(script_directory, '..', dir_name)
# Create the output directory if it doesn't exist
if not os.path.exists(write_data_directory):
    os.makedirs(write_data_directory)

# Dynamic directory paths
x_df_path = os.path.join(read_data_directory, 'x_df.pkl')
time_df_path = os.path.join(read_data_directory, 'time_series_data.pkl')
census_df_path = os.path.join(read_data_directory, 'fpop_final.pkl')
fpop_connections_path = os.path.join(read_data_directory, 'fpop_connections_dict.pkl')
fpop_reverse_connections_path = os.path.join(read_data_directory, 'fpop_reverse_connections_dict.pkl')

x_df = pd.read_pickle(x_df_path) # x_df - the quantified IFEM component scores
time_df = pd.read_pickle(time_df_path) # Read in time-series data (news, price history, BEV registrations)
census_df = pd.read_pickle(census_df_path) # Synthetic Census Data

# Social Network Dicts
with open(fpop_connections_path, 'rb') as f:
    connections_dict = pickle.load(f)
with open(fpop_reverse_connections_path, 'rb') as f:
    reverse_connections_dict = pickle.load(f)

if sample == True:
    n = 90000  # Set the number of rows to iterate over
    # Get n random rows from the DataFrame
    sample = census_df.sample(n)
else:
    sample = census_df

# Sort the dataframe by 'hhold' and 'htype'
census_sorted = sample.sort_values(by=['hhold', 'htype'])
# Group the dataframe by 'hhold' and 'htype'
grouped_census = census_sorted.groupby(['hhold', 'htype'])

annual_replacement_percents = [0.0409, 0.0462, 0.0503, 0.0569, 0.0607, 0.0632, 0.0660, 0.0650, 0.0629, 0.0627, 
                               0.0623, 0.0615, 0.0513, 0.0485] # 2009 - 2022

charger_estimate = [1, 1, 7, 19, 25, 32, 43, 53, 60, 72, 83, 99, 157, 181, 217] # 2009 - 2023 based on Fairfax Population

years = list(range(20))

class Consumer(mesa.Agent):
    """
    Possible EV consumer agent
    """
    def __init__(self, pos, model, agent_type, age, charge_inf, education, experience, hh_size, housing, htype, income, kids, local_inf, politics, sex, shopping, social_inf, weighted_sum):
        """
        Create a new Consumer agent.
        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.age = age
        self.charge_inf = charge_inf
        self.education = education
        self.experience = experience
        self.hh_size = hh_size
        self.housing = housing
        self.htype = htype
        self.income = income
        self.kids = kids
        self.local_inf = local_inf
        self.politics = politics
        self.sex = sex
        self.shopping = shopping
        self.social_inf = social_inf
        self.weighted_sum = weighted_sum

    def threshold_function(self, x_df):

        g = 0.75 if self.sex == 'm' else 0.25 # Set sex score
        pol = 0.75 if self.politics == 'D' else 0.25 # Set political score
        # set housing score
        if self.housing == 'Single Family':
            h = 0.75
        elif self.housing == 'Duplex' or self.housing == 'Townhouse':
            h = 0.50
        elif self.housing == 'Multiplex' or self.housing == 'Multifamily':
            h = 0.25

        pe_value = ((1 - self.age/100) + g) * x_df.at[0,'performance_expectancy']
        dp_value = ((1 - self.age/100) - self.experience + g) * x_df.at[0,'driving_pleasure'] 
        tr_value = self.charge_inf * x_df.at[0,'translation']
        si_value = (self.local_inf + self.social_inf) * x_df.at[0,'social_influence']
        is_value = self.education * x_df.at[0,'identity_symbolism']
        en_value = (pol + time_df.iloc[self.model.tick - start_tick,8]) * x_df.at[0,'enrollment'] # time_df 8 is smoothed_norm news
        fc_value = (self.charge_inf + h) * x_df.at[0,'facilitating_conditions'] #
        pr_value = ((self.age/100) + g + self.income + time_df.iloc[self.model.tick - start_tick,6]) * x_df.at[0,'price_value'] # time_df 6 is price history
        eh_value = self.experience * x_df.at[0,'experience_and_habit']

        pe_value = pe_value * self.model.ifem_pe_var
        dp_value = dp_value * self.model.ifem_dp_var
        tr_value = tr_value * self.model.ifem_tr_var
        si_value = si_value * self.model.ifem_si_var
        is_value = is_value * self.model.ifem_is_var
        en_value = en_value * self.model.ifem_en_var
        fc_value = fc_value * self.model.ifem_fc_var
        pr_value = pr_value * self.model.ifem_pr_var
        eh_value = eh_value * self.model.ifem_eh_var

        weighted_sum = pe_value + dp_value + tr_value + si_value + is_value + en_value + fc_value + pr_value + eh_value

        return weighted_sum

    def step(self):

        # Calculate BEV experience
        if self.type == 'bev_owner':
            if self.experience <= 0.9: # Upper limit for testing
                self.experience += 0.1

        if self.shopping == True: # Begin agent behavior with burn-in complete (commencing May 2014)

            # Social network influence calculation
            self.social_inf = 0  

            # Retrieve the IDs of agents in the current agent's social network
            social_network = connections_dict.get(self.unique_id, [])

            # Iterate through the agents in the social network
            for connection_id in social_network:
              
                if connection_id in self.model.loaded_ids:
                    
                    # Check if the agent in the social network is a 'bev_owner'
                    if reverse_connections_dict.get(connection_id, None) and self.model.schedule._agents[connection_id].type == 'bev_owner':
                        self.social_inf += 1 # Increment the social influence score

            if self.social_inf > 3:
                self.social_inf = 3

            # BEV owning neighbor proximity influence
            for neighbor in self.model.grid.iter_neighbors(self.pos, True):
                self.local_inf = 0
                if neighbor.type == 'bev_owner':
                    self.local_inf += 0.125 # Up to 1 for all 8 neighbors
                if self.local_inf > 3:
                    self.local_inf = 3 # Don't let it grow ridiculously in multigrid

            # Charger proximity influence
            if self.charge_inf < 3.0:
                for neighbor in self.model.grid.iter_neighbors(self.pos, True, radius=2):
                    if neighbor.type == 'charger' and neighbor.status == 'on':
                        self.charge_inf += 0.5

            self.weighted_sum = self.threshold_function(x_df)

            # Determine if new car is BEV
            if self.weighted_sum >= self.model.bev_thresh and self.type != 'bev_owner':
                self.type = 'bev_owner'
                self.model.bevs += 1
            if self.weighted_sum < self.model.bev_thresh and self.type == 'bev_owner':
                self.type = 'non_bev_owner'
                self.model.bevs -= 1

            self.shopping = False # Regardless of purchase type, no longer shopping

class ChargingAgent(mesa.Agent):
    """
    Charging location agent
    """
    def __init__(self, pos, model, agent_type, status):
        """
        Create a new charging agent.

        Args:
           unique_id: Unique identifier for the agent.
           x, y: Agent initial location.
        """
        super().__init__(pos, model)
        self.pos = pos
        self.type = agent_type
        self.status = status
    
    def step(self):
        
        num_chargers = charger_estimate[self.model.tick // 12] * (self.model.total_agents / 506959)

        if self.model.on_chargers < num_chargers and self.status == 'off':
            self.status = 'on'
            self.model.on_chargers +=1

class FairfaxABM(mesa.Model):
    """
    Model class for the BEV adoption model.
    """
    
    def __init__(self, width=30, height=30, rand_or_gis=0.0, bev_thresh=1, total_agents=0, bevs=0, bev_target=0, 
        tick=1, loaded_ids=[], charge_count=0, on_chargers=0, pe_var=1, dp_var=1, tr_var=1, si_var=1, is_var=1, en_var=1, fc_var=1,
        pr_var=1, eh_var=1, agent_df=None):
        """ """
        
        self.width = width
        self.height = height
        self.rand_or_gis = rand_or_gis
        self.bev_thresh = bev_thresh
        self.total_agents = total_agents
        self.bevs = bevs
        self.bev_target = bev_target
        self.tick = tick
        self.loaded_ids = loaded_ids
        self.charge_count = charge_count
        self.on_chargers = on_chargers

        # Tuning variables
        self.ifem_pe_var = pe_var
        self.ifem_dp_var = dp_var
        self.ifem_tr_var = tr_var
        self.ifem_si_var = si_var
        self.ifem_is_var = is_var
        self.ifem_en_var = en_var
        self.ifem_fc_var = fc_var
        self.ifem_pr_var = pr_var
        self.ifem_eh_var = eh_var

        self.agent_df = pd.DataFrame(columns=['step', 'unique_id', 'type', 'age', 'charge_inf', 'education', 'experience', 'hh_size', 'housing', 'htype', 'income', 'kids', 'local_inf', 'politics', 'sex', 'shopping', 'social_inf', 'weighted_sum'])

        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(width, height, torus=False)

        self.datacollector = mesa.DataCollector(
            {"bevs": "bevs",  # Model-level count of BEV owners
            "bev_target": "bev_target",
            "on_chargers": "on_chargers",},)
            # For testing purposes, agent's individual x and y, expands df size substantially (7k to 96mb for 10k run)
            #{"bev_owner": lambda a: a.type},)#, "age": lambda a: a.age},)

        def placement_funct():
            """
            Function that handles operations applicable to all consumer agents
            """
            age = row['age']
            education = row['education']
            income = row['income']
            housing = row['housing']
            politics = row['political_orientation']
            sex = row['sex']
            shopping = False
            unique_id = row['individual_id']

            if self.rand_or_gis == 0: # Load agents randomly for rand_or_gis == 0  
                    
                x = self.random.randrange(self.grid.width) # added for MultiGrid and random placement
                y = self.random.randrange(self.grid.height)

            if self.rand_or_gis == 1: # Load agents by normalized lat/lon for rand_or_gis == 1
            
                # Extract normalized latitude and longitude from the selected row
                normalized_lat = row['NormalizedLat']
                normalized_lon = row['NormalizedLon']

                # Convert normalized coordinates to grid coordinates
                x = int(normalized_lat * self.grid.width)
                y = int(normalized_lon * self.grid.height)

                # Ensure the coordinates are within bounds
                x = max(0, min(x, self.grid.width - 1))
                y = max(0, min(y, self.grid.height - 1))

            agent_type = 'non_bev_owner'

            # This is the percentage registerred for May 2014, first BEVs in Fairfax per data. However, things start in April. Hence all initially non-BEV
            #if self.random.random() < 0.000219: agent_type = 'bev_owner'
            #    self.bevs+=1
            #else:
            #    agent_type = 'non_bev_owner'

            self.total_agents += 1

            weighted_sum = 0 # Initially give all agents a 0

            agent = Consumer((unique_id), self, agent_type, age, charge_inf, education, experience, hh_size, housing, htype, income, kids, local_inf, politics, sex, shopping, social_inf, weighted_sum) # Needed for multigrid but should lilely stay regardless for individual agent tracking
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
        
        charge_inf = 0
        experience = 0
        local_inf = 0
        social_inf = 0

        for (hhold, htype), group in grouped_census:
            # Access the inhabitants of the current household group

            inhabitants = group[['individual_id', 'age', 'sex', 'hhold', 'housing', 'htype', 'education', 'income', 'political_orientation', 'NormalizedLat', 'NormalizedLon']]
            inhab_sorted = inhabitants.sort_values(by=['age', 'sex'], ascending=[False, True])

            hh_size = len(inhabitants) # Set size of household

            if htype in (0,2,4,6,7,8,9,10):
                kids = False
            if htype in (1,3,5):
                kids = True

            if htype in (0, 1): # husband and wife
                try:
                    index_num = inhab_sorted[inhab_sorted['sex'] == 'm'].index[0] # Husband
                    row = inhab_sorted.loc[index_num]
                    if row['age'] >= 18:
                        placement_funct()
                except:
                    None

                try:
                    index_num = inhab_sorted[inhab_sorted['sex'] == 'f'].index[0] # Wife
                    row = inhab_sorted.loc[index_num]
                    if row['age'] >= 18:
                        placement_funct()
                except:
                    None

            if htype in (2,3): # Male with and without kids
                for index, row in inhab_sorted.iterrows():
                    if row['age'] >= 18:
                        if row['sex'] == 'm':
                            placement_funct()

            if htype in (4,5): # female with no / with kids
                try:
                    index_num = inhab_sorted[inhab_sorted['sex'] == 'f'].index[0]
                    row = inhab_sorted.loc[index_num]
                    if row['age'] >= 18:
                        placement_funct()
                except:
                    None

            if htype == 6:
                for index, row in inhab_sorted.iterrows():
                    if row['age'] >= 18:
                        placement_funct()

            if htype in (7,8,9,10):
                for index, row in inhab_sorted.iterrows():
                    if row['age'] >= 18:
                        placement_funct()

        # Load chargers
        num_chargers = charger_estimate[14] * (self.total_agents / 506959) # 506,959 is total population load

        # Load in Chargers
        while self.charge_count < num_chargers:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)

            charge_id = f'c{self.charge_count}'  # Generate charge ID
            agent_type = 'charger'
            status = 'off'
            agent = ChargingAgent((charge_id), self, agent_type, status)

            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)
            self.charge_count += 1

        # Get the set of consumer agent IDs currently loaded in the model
        loaded_agent_ids = {agent.unique_id for agent in self.schedule.agents if agent.type == 'bev_owner' or agent.type == 'non_bev_owner'}

        self.loaded_ids = loaded_agent_ids

        print('Total agents:',self.total_agents)
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model. If All agents are BEV owners, halt the model.
        """

        # Adjust the BEV threshold over time to reduce exponential growth

        threshold_adjustment = 0.1*(np.log1p(self.tick) / 100)  # Or use a sigmoid as described above
        self.bev_thresh += threshold_adjustment

        # For agent data capture
        step_data = []

        # Perform car age transformation
        year = (self.tick // 12)

        # Determine the number of agents to shop
        n = round((annual_replacement_percents[year] / 12) * self.total_agents)

        # Shuffle the list of agents and select the first `n` agents to shop
        agents_to_shop = random.sample([a for a in self.schedule.agents if a.type in ['bev_owner', 'non_bev_owner']], n)

        # Set the selected agents to shop
        for a in agents_to_shop:
            if self.tick >= start_tick:
                a.shopping = True   

        #for a in agents_to_shop[:5]:
        #    print('Shopping agents:', a.unique_id)

        # Single agent schedule loop
        if agent_data == True:

            # Agent level data capture
            for a in self.schedule.agents:

                if self.tick == 1:
                    if a.type != 'charger':
                        step_data.append({
                            'step': self.tick,
                            'unique_id': a.unique_id,
                            'type': a.type,
                            'age': a.age,
                            'charge_inf': a.charge_inf,
                            'education': a.education,
                            'experience': a.experience,
                            'hh_size': a.hh_size,
                            'housing': a.housing,
                            'htype': a.htype,
                            'income': a.income,
                            'kids': a.kids,
                            'local_inf': a.local_inf,
                            'politics': a.politics,
                            'sex': a.sex,
                            'shopping': a.shopping,
                            'social_inf': a.social_inf,
                            'weighted_sum': a.weighted_sum})

                if a.type == 'bev_owner':
                    step_data.append({
                        'step': self.tick,
                        'unique_id': a.unique_id,
                        'type': a.type,
                        'age': a.age,
                        'charge_inf': a.charge_inf,
                        'education': a.education,
                        'experience': a.experience,
                        'hh_size': a.hh_size,
                        'housing': a.housing,
                        'htype': a.htype,
                        'income': a.income,
                        'kids': a.kids,
                        'local_inf': a.local_inf,
                        'politics': a.politics,
                        'sex': a.sex,
                        'shopping': a.shopping,
                        'social_inf': a.social_inf,
                        'weighted_sum': a.weighted_sum})
        
        # Capture agent level data
        step_df = pd.DataFrame(step_data)
        self.agent_df = pd.concat([self.agent_df, step_df], ignore_index=True)


        ### OUTPUT CAPTURE AND DUMP FOR ANALYSIS ###
        
        if self.tick >= start_tick: # self.tick 65 is the first tick with a percent of BEVs

            #print('Tick:',self.tick)
            print(time_df.iloc[(self.tick-49),0])

            historic_percent = round((time_df.iloc[(self.tick - 49), 7]),6) # 7 is smoothed column # was 49
            self.bev_target = round((historic_percent * self.total_agents),0)

            self.schedule.step()
            # collect data
            self.datacollector.collect(self)

            if self.tick == 167: # Halt when registrations end (2022)

                df_file_path = os.path.join(write_data_directory, 'agent_data.pkl')
                self.agent_df.to_pickle(df_file_path)

                self.running = False

        else:
            
            self.schedule.step()
            # collect data
            self.datacollector.collect(self)

        self.tick+=1

# Initial params
#params = {"width": 30, "height": 30, "rand_or_gis": [0.0], "bev_thresh": [3.5], "total_agents": [0], "tick": [1], 'loaded_ids': [[]], 'charge_count': [0], 'on_chargers': [0], 
#    'pe_var': [1], 'dp_var': [1], 'tr_var': [1], 'si_var': [1], 'is_var': [1], 'en_var': [1], 'fc_var': [1], 'pr_var': [1], 'eh_var': [1]} 

# Test params
#params = {"width": 30, "height": 30, "rand_or_gis": [0.0], "bev_thresh": [3.5], "total_agents": [0], "tick": [1], 'loaded_ids': [[]], 'charge_count': [0], 'on_chargers': [0], 
#    'pe_var': [1.25], 'dp_var': [1.25], 'tr_var': [1.0], 'si_var': [0.5], 'is_var': [1.2], 'en_var': [.75], 'fc_var': [0.5], 'pr_var': [0.5], 'eh_var': [1.1]} 

# Sensitivity params
params = {"width": 30, "height": 30, "rand_or_gis": [0.0], "bev_thresh": [3.5], "total_agents": [0], "tick": [1], 'loaded_ids': [[]], 'charge_count': [0], 'on_chargers': [0], 
    'pe_var': [1.25], 'dp_var': [1.25], 'tr_var': [1.0], 'si_var': [0.5], 'is_var': [1.2], 'en_var': [.75], 'fc_var': [0.5], 'pr_var': [0.5], 'eh_var': [1.1]} 

# Best Params
#params = {"width": 30, "height": 30, "rand_or_gis": [0.0], "bev_thresh": [3.5], "total_agents": [0], "tick": [1], 'loaded_ids': [[]], 'charge_count': [0], 'on_chargers': [0], 
 #   'pe_var': [1.25], 'dp_var': [1.25], 'tr_var': [1.0], 'si_var': [0.5], 'is_var': [1.2], 'en_var': [.75], 'fc_var': [0.5], 'pr_var': [0.5], 'eh_var': [1.1]} 

if __name__ == "__main__":

    start = time.time()

    results = mesa.batch_run(
    FairfaxABM,
    parameters=params,
    iterations=8,
    max_steps=167,
    number_processes=None,
    data_collection_period=1,
    display_progress=True)

    results_df = pd.DataFrame(results)

    df = results_df.drop(['width', 'height', 'total_agents', 'tick', 'loaded_ids', 'charge_count'], axis=1)

    csv_file_path = os.path.join(write_data_directory, 'FairfaxABM_Data.csv')
    df.to_csv(csv_file_path)

    end = time.time()

    print('Process took', round((end-start),2), 'seconds.')



