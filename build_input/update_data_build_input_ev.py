# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:35:09 2019

@author: arvensyla
"""

# execute primary input data building script
# import build_input_dsr_ee_dhw
print('####################')
print('BUILDING INPUT DATA FOR Electric Vehicles')
print('####################')
import sys, os
import os
import itertools
import hashlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
import datetime

import seaborn as sns
from grimsel.auxiliary.aux_general import print_full
from grimsel.auxiliary.aux_general import translate_id

import config_local as conf
from grimsel.auxiliary.aux_general import expand_rows

base_dir = conf.BASE_DIR
data_path = conf.PATH_CSV + '\\csv_files_new_ev'
data_path_prv = conf.PATH_CSV + '\\wo_update_input_data_ev'
seed = 2

np.random.seed(seed)

db = conf.DATABASE
sc = conf.SCHEMA

def append_new_rows(df, tb):

    list_col = list(aql.get_sql_cols(tb, sc, db).keys())

    aql.write_sql(df[list_col], db=db, sc=sc, tb=tb, if_exists='append')

def del_new_rows(ind, tb, df):

    del_list = df[ind].drop_duplicates()

    for i in ind:
        del_list[i] = '%s = '%i + del_list[i].astype(str)

    del_str = ' OR '.join(del_list.apply(lambda x: '(' + ' AND '.join(x) + ')', axis=1))

    exec_strg = '''
                DELETE FROM {sc}.{tb}
                WHERE {del_str}
                '''.format(tb=tb, sc=sc, del_str=del_str)
    aql.exec_sql(exec_strg, db=db)

def replace_table(df, tb):
    print('Replace table %s'%tb)
#    list_col = list(aql.get_sql_cols(tb, sc, db).keys())
    
    aql.write_sql(df, db=db, sc=sc, tb=tb, if_exists='replace')
 

def append_new_cols(df, tb):
#
    list_col = list(aql.get_sql_cols(tb, sc, db).keys())
    
    col_new = dict.fromkeys((set(df.columns.tolist()) - set(list_col)))
    for key, value in col_new.items():
        col_new[key] = 'DOUBLE PRECISION'
#    col_new = dict.fromkeys((set(list_col[0].columns.tolist()) - set(list_col)),1)
    
    aql.add_column(df_src=df,tb_tgt=[sc,tb],col_new=col_new,on_cols=list_col, db=db)



# %% ~~~~~~~~~~~~~~~~~~   DEF_NODE (we update def_node - new CO2 carbon taxes)

df_def_node_0 = pd.read_csv(data_path_prv + '\\def_node.csv')

new_values = {
    'price_co2_yr2030': 113.4,
    'price_co2_yr2035': 130.2,
    'price_co2_yr2040': 147,
    'price_co2_yr2045': 157.5,
    'price_co2_yr2050': 168,
}

df_def_node_0.loc[:4, new_values.keys()] = new_values.values()


df_def_node_0.to_csv(data_path + '\\def_node.csv', index=False)

# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PP_TYPE

df_def_pp_type_0 = pd.read_csv(data_path_prv + '\\def_pp_type.csv')

df_def_pp_type = df_def_pp_type_0.copy().head(0)

for npt, pt, cat, color in ((0, 'STO_LI', 'NEW_STORAGE', '#ff8000'),
                            ):

    df_def_pp_type.loc[npt] = (npt, pt, cat, color)


df_def_pp_type.loc[:,'pt_id'] = np.arange(0, len(df_def_pp_type)) + df_def_pp_type_0.pt_id.max() + 1

# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_FUEL for the update of parameters

# ---> NO CHANGES

# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_ENCAR for the update of parameters

#  ----> NO CHANGES

# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PLANT

df_def_plant_0 = pd.read_csv(data_path + '/def_plant.csv')


df_def_plant = df_def_plant_0.copy().head(0)

for npt, pp, nd_id, fl_id, pt_id, set_def_pr, set_def_cain, set_def_ror, set_def_pp, set_def_st, set_def_hyrs, set_def_chp, set_def_add, set_def_rem, set_def_sll, set_def_curt, set_def_lin, set_def_scen, set_def_winsol, set_def_tr, set_def_peak in (
        (0,'IT_WIN_OFF', 4,	4,	3,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0, 0),
        (1, 'AT_STO_LI', 0,	19,	(np.arange(0, len(df_def_pp_type)) + df_def_pp_type_0.pt_id.max() + 1)[0],	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0),
        (2, 'DE_STO_LI', 2,	19,	(np.arange(0, len(df_def_pp_type)) + df_def_pp_type_0.pt_id.max() + 1)[0],	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0),
        (3, 'FR_STO_LI', 3,	19,	(np.arange(0, len(df_def_pp_type)) + df_def_pp_type_0.pt_id.max() + 1)[0],	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0),
        (4, 'IT_STO_LI', 4,	19,	(np.arange(0, len(df_def_pp_type)) + df_def_pp_type_0.pt_id.max() + 1)[0],	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0),
        (5, 'IND_RUR_DMND_FLEX', 5, 23, 16,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (6, 'IND_SUB_DMND_FLEX', 6, 23, 16,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (7, 'IND_URB_DMND_FLEX', 7, 23, 16,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (8, 'MFH_RUR_0_DMND_FLEX', 8, 23, 16,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (9, 'MFH_RUR_1_DMND_FLEX', 9, 23, 16,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (10, 'MFH_RUR_2_DMND_FLEX', 10, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (11, 'MFH_RUR_3_DMND_FLEX', 11, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (12, 'MFH_SUB_0_DMND_FLEX', 12, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (13, 'MFH_SUB_1_DMND_FLEX', 13, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (14, 'MFH_SUB_2_DMND_FLEX', 14, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (15, 'MFH_SUB_3_DMND_FLEX', 15, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (16, 'MFH_URB_0_DMND_FLEX', 16, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (17, 'MFH_URB_1_DMND_FLEX', 17, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (18, 'MFH_URB_2_DMND_FLEX', 18, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (19, 'MFH_URB_3_DMND_FLEX', 19, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (20, 'OCO_RUR_DMND_FLEX', 20, 23, 16, 0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (21, 'OCO_SUB_DMND_FLEX', 21, 23, 16, 0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (22, 'OCO_URB_DMND_FLEX', 22, 23, 16, 0, 0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (23, 'SFH_RUR_0_DMND_FLEX', 23, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (24, 'SFH_RUR_1_DMND_FLEX', 24, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (25, 'SFH_RUR_2_DMND_FLEX', 25, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (26, 'SFH_RUR_3_DMND_FLEX', 26, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (27, 'SFH_SUB_0_DMND_FLEX', 27, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (28, 'SFH_SUB_1_DMND_FLEX', 28, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (29, 'SFH_SUB_2_DMND_FLEX', 29, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (30, 'SFH_SUB_3_DMND_FLEX', 30, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (31, 'SFH_URB_0_DMND_FLEX', 31, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (32, 'SFH_URB_1_DMND_FLEX', 32, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (33, 'SFH_URB_2_DMND_FLEX', 33, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        (34, 'SFH_URB_3_DMND_FLEX', 34, 23, 16, 0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0),
        ): 

    df_def_plant.loc[npt] = (npt, pp, nd_id, fl_id, pt_id, set_def_pr, set_def_cain, 
                             set_def_ror, set_def_pp, set_def_st, set_def_hyrs, 
                             set_def_chp, set_def_add, set_def_rem, set_def_sll, 
                             set_def_curt, set_def_lin, set_def_scen, set_def_winsol, 
                             set_def_tr, set_def_peak)




df_def_plant.loc[:,'pp_id'] = np.arange(0, len(df_def_plant)) + df_def_plant_0.pp_id.max() + 1
                   #pp_id
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEF_PROFILE for EV

df_def_profile_0 = pd.read_csv(data_path_prv + '\\def_profile.csv')


df_def_profile_wind_offshore_it = df_def_plant[df_def_plant.pp.str.contains('IT_WIN')].pp.copy().rename('primary_nd').reset_index()
df_def_profile_wind_offshore_it ['pf'] = 'supply_' + df_def_profile_wind_offshore_it.primary_nd
df_def_profile_wind_offshore_it ['pf_id'] = df_def_profile_wind_offshore_it.index.rename('pf_id') + df_def_profile_0.pf_id.max() + 1
df_def_profile_wind_offshore_it = df_def_profile_wind_offshore_it[df_def_profile_0.columns]


df_def_profile = pd.concat([df_def_profile_wind_offshore_it,], axis=0)
df_def_profile = df_def_profile.reset_index(drop=True)
    
df_def_profile

# %% ~~~~~~~~~  NODE_ENCAR for new updates --- only electricity demand is increased on neighbouring countries

df_node_encar_0 = pd.read_csv(data_path_prv + '/node_encar.csv')

df_node_encar = df_node_encar_0.copy()

new_values_for_id_0 = { 'dmnd_sum_yr2030': 72506178.76, 'dmnd_sum_yr2035': 77048730.13, 'dmnd_sum_yr2040': 81591281.5,
    'dmnd_sum_yr2045': 86133832.87, 'dmnd_sum_yr2050': 90676384.24}

# Update the DataFrame based on nd_id value
df_node_encar.loc[df_node_encar['nd_id'] == 0, new_values_for_id_0.keys()] = new_values_for_id_0.values()
# For example, for nd_id == 2 // DE
new_values_for_id_2 = {'dmnd_sum_yr2030': 651500000, 'dmnd_sum_yr2035': 685910000, 'dmnd_sum_yr2040': 720320000,
    'dmnd_sum_yr2045': 745845000, 'dmnd_sum_yr2050': 771370000}
df_node_encar.loc[df_node_encar['nd_id'] == 2, new_values_for_id_2.keys()] = new_values_for_id_2.values()

# For example, for nd_id == 3 // FR
new_values_for_id_3 = { 'dmnd_sum_yr2030': 523310000, 'dmnd_sum_yr2035': 546925000, 'dmnd_sum_yr2040': 570540000,
    'dmnd_sum_yr2045': 584015000, 'dmnd_sum_yr2050': 597490000}
df_node_encar.loc[df_node_encar['nd_id'] == 3, new_values_for_id_3.keys()] = new_values_for_id_3.values()

# Repeat the process for other nd_id values with their respective new values
# For example, for nd_id == 4 // IT
new_values_for_id_4 = { 'dmnd_sum_yr2030': 325640000, 'dmnd_sum_yr2035': 350279591.4, 'dmnd_sum_yr2040': 374919182.9,
    'dmnd_sum_yr2045': 392063853.8, 'dmnd_sum_yr2050': 409208524.6}
df_node_encar.loc[df_node_encar['nd_id'] == 4, new_values_for_id_4.keys()] = new_values_for_id_4.values()


df_node_encar.to_csv(data_path + '\\node_encar.csv', index=False)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFDMND for EV

# --> NO CHANGES! 
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFPRICE

# --> NO CHANGES! 
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFSUPPLY
# 
df_profsupply_0 = pd.read_csv(data_path_prv + '/profsupply.csv')

df_wind_offshore_it_supply =  pd.read_csv(base_dir + '\\update_data_with_recent_scenarios\\profsupply_it_windoffshore_new.csv')


dict_pf_id = df_def_profile.set_index('pf')['pf_id'].to_dict()


dict_pf_id = {'supply_IT_WIN_OFF': dict_pf_id['supply_' + 'IT' + '_WIN_OFF']}
df_wind_offshore_it_supply['supply_pf_id'] = df_wind_offshore_it_supply.supply_pf_id.replace(dict_pf_id)

df_wind_offshore_it_supply['supply_pf_id'] = dict_pf_id['supply_IT_WIN_OFF']

df_profsupply = df_wind_offshore_it_supply[df_wind_offshore_it_supply.columns.tolist()]

# %% ~~~~~~~~~~~~~~~~~~~~~~~ PLANT_ENCAR 
# Update the values of plant capacities/add plants with new capacities!!!!

df_plant_encar_0 = pd.read_csv(data_path_prv + '/plant_encar.csv')

# First, we replace the previous values with the new scenarios

df_plant_encar_0

new_values_for_pp_id_10 = { 'cap_pwr_leg_yr2030': 2902.366092, 'cap_pwr_leg_yr2035': 957, 'cap_pwr_leg_yr2040': 957,
    'cap_pwr_leg_yr2045': 957, 'cap_pwr_leg_yr2050': 957, 'erg_chp_yr2030':3800089.65, 'erg_chp_yr2035': 950022.41, 
    'erg_chp_yr2040': 950022.41, 'erg_chp_yr2045': 950022.41, 'erg_chp_yr2050': 950022.41}

# Update the DataFrame based on nd_id value
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 10, new_values_for_pp_id_10.keys()] = new_values_for_pp_id_10.values()
# For example, for nd_id == 2 // DE

new_values_for_pp_id_12 = { 'cap_pwr_leg_yr2040': 19627.07, 'cap_pwr_leg_yr2045': 15791.96, 'cap_pwr_leg_yr2050': 12869.26,
                           'erg_chp_yr2030':33274673.58, 'erg_chp_yr2035': 13134739.57, 
    'erg_chp_yr2040': 0, 'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 12, new_values_for_pp_id_12.keys()] = new_values_for_pp_id_12.values()
##
new_values_for_pp_id_13 = {'cap_pwr_leg_yr2030': 7673,'cap_pwr_leg_yr2035': 6209, 
                           'cap_pwr_leg_yr2040': 5577, 'cap_pwr_leg_yr2045': 2739, 
                           'cap_pwr_leg_yr2050': 697, 'erg_chp_yr2030':4781119.17, 
                           'erg_chp_yr2035': 4549355.50, 'erg_chp_yr2040': 4163139.60, 
                           'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 13, new_values_for_pp_id_13.keys()] = new_values_for_pp_id_13.values()

##
new_values_for_pp_id_14 = {'erg_chp_yr2030':21517276.72, 'erg_chp_yr2035': 14344851.15, 
                           'erg_chp_yr2040': 8068978.77, 'erg_chp_yr2045': 3586212.79, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 14, new_values_for_pp_id_14.keys()] = new_values_for_pp_id_14.values()

##
new_values_for_pp_id_16 = {'cap_pwr_leg_yr2030': 7673,'cap_pwr_leg_yr2035': 6209, 
                           'cap_pwr_leg_yr2040': 5577, 'cap_pwr_leg_yr2045': 2739, 
                           'cap_pwr_leg_yr2050': 697, 'erg_chp_yr2030':2445300, 
                           'erg_chp_yr2035': 2445300, 'erg_chp_yr2040': 2445300, 
                           'erg_chp_yr2045': 2445300, 'erg_chp_yr2050': 2445300}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 16, new_values_for_pp_id_16.keys()] = new_values_for_pp_id_16.values()

##
new_values_for_pp_id_17 = {'erg_chp_yr2025':1493436.28, 'erg_chp_yr2030':6400800.32, 'erg_chp_yr2035': 21637044.29, 
                           'erg_chp_yr2040': 11171896.2, 'erg_chp_yr2045': 11171896.2, 'erg_chp_yr2050': 10302970.94}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 17, new_values_for_pp_id_17.keys()] = new_values_for_pp_id_17.values()

##
new_values_for_pp_id_18 = {'cap_pwr_leg_yr2030': 0,'cap_pwr_leg_yr2035': 0, 
                           'cap_pwr_leg_yr2040': 0, 'cap_pwr_leg_yr2045': 0, 
                           'cap_pwr_leg_yr2050': 0, 'erg_chp_yr2030':0, 
                           'erg_chp_yr2035': 0, 'erg_chp_yr2040': 0, 
                           'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 18, new_values_for_pp_id_18.keys()] = new_values_for_pp_id_18.values()

##
new_values_for_pp_id_19 = {'erg_chp_yr2030': 15364375.03, 'erg_chp_yr2035': 15364375.03, 'erg_chp_yr2040': 15364375.03, 
                           'erg_chp_yr2045': 15148180.58, 'erg_chp_yr2050': 11039404.8}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 19, new_values_for_pp_id_19.keys()] = new_values_for_pp_id_19.values()

##
new_values_for_pp_id_20 = {'cap_pwr_leg_yr2025': 0, 'cap_pwr_leg_yr2030': 0,'cap_pwr_leg_yr2035': 0, 
                           'cap_pwr_leg_yr2040': 0, 'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2025':0, 'erg_chp_yr2030':0, 'erg_chp_yr2035': 0, 
                           'erg_chp_yr2040': 0, 'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 20, new_values_for_pp_id_20.keys()] = new_values_for_pp_id_20.values()

##
new_values_for_pp_id_21 = {'cap_pwr_leg_yr2030': 7851.9,'cap_pwr_leg_yr2035': 0, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2025':7134777.65, 'erg_chp_yr2030':438205.51, 'erg_chp_yr2035': 0, 
                           'erg_chp_yr2040': 0, 'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 21, new_values_for_pp_id_21.keys()] = new_values_for_pp_id_21.values()

##
new_values_for_pp_id_22 = {'cap_pwr_leg_yr2030': 1740,'cap_pwr_leg_yr2035': 0, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2025':965445.56, 'erg_chp_yr2030':2597959.56, 'erg_chp_yr2035': 0, 
                           'erg_chp_yr2040': 0, 'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 22, new_values_for_pp_id_22.keys()] = new_values_for_pp_id_22.values()

##
new_values_for_pp_id_23 = {'cap_pwr_leg_yr2030': 1048,'cap_pwr_leg_yr2035': 1048, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2025':65803.71, 'erg_chp_yr2030':13512.75, 'erg_chp_yr2035': 13512.75, 
                           'erg_chp_yr2040': 0, 'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 23, new_values_for_pp_id_23.keys()] = new_values_for_pp_id_23.values()

##
new_values_for_pp_id_24 = {'cap_pwr_leg_yr2030': 9190,'cap_pwr_leg_yr2035': 5892, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2025':3526651.93, 'erg_chp_yr2030':1927764.09, 'erg_chp_yr2035': 963882.045, 
                           'erg_chp_yr2040': 0, 'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0}
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 24, new_values_for_pp_id_24.keys()] = new_values_for_pp_id_24.values()


##
new_values_for_pp_id_25 = {'cap_pwr_leg_yr2030': 115000.7,'cap_pwr_leg_yr2035': 136939.5, 'cap_pwr_leg_yr2040': 158878.29, 
                           'cap_pwr_leg_yr2045': 160003.91, 'cap_pwr_leg_yr2050': 161129.52, 
                           'fc_om_yr2030': 12600, 'fc_om_yr2035': 12965.4, 'fc_om_yr2040': 13330.8, 
                           'fc_om_yr2045': 13620.6, 'fc_om_yr2050': 13910.4 
                           }
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 25, new_values_for_pp_id_25.keys()] = new_values_for_pp_id_25.values()

##
new_values_for_pp_id_26 = {'cap_pwr_leg_yr2030': 7911.38,'cap_pwr_leg_yr2035': 9418.32, 'cap_pwr_leg_yr2040': 10925.25, 
                           'cap_pwr_leg_yr2045': 13462.62, 'cap_pwr_leg_yr2050': 16000, 
                           'fc_om_yr2030': 12600, 'fc_om_yr2035': 12965.4, 'fc_om_yr2040': 13330.8, 
                           'fc_om_yr2045': 13620.6, 'fc_om_yr2050': 13910.4 
                           }
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 26, new_values_for_pp_id_26.keys()] = new_values_for_pp_id_26.values()

##
new_values_for_pp_id_27 = {'cap_pwr_leg_yr2030': 18410.82,'cap_pwr_leg_yr2035': 20738.32, 'cap_pwr_leg_yr2040': 23065.82, 
                           'cap_pwr_leg_yr2045': 26522.52, 'cap_pwr_leg_yr2050': 29979.23, 
                           'fc_om_yr2030': 12600, 'fc_om_yr2035': 12965.4, 'fc_om_yr2040': 13330.8, 
                           'fc_om_yr2045': 13620.6, 'fc_om_yr2050': 13910.4 
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 27, new_values_for_pp_id_27.keys()] = new_values_for_pp_id_27.values()

##
new_values_for_pp_id_28 = {'fc_om_yr2030': 55764.8, 'fc_om_yr2035': 57381.98, 'fc_om_yr2040': 58999.16, 
                           'fc_om_yr2045': 60281.75, 'fc_om_yr2050': 61564.34 
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 28, new_values_for_pp_id_28.keys()] = new_values_for_pp_id_28.values()

##
new_values_for_pp_id_29 = {'cap_pwr_leg_yr2030': 35959,'cap_pwr_leg_yr2035': 41519.45, 'cap_pwr_leg_yr2040': 47079.9, 
                           'cap_pwr_leg_yr2045': 52944.95, 'cap_pwr_leg_yr2050': 58810, 
                           'fc_om_yr2030': 12600, 'fc_om_yr2035': 12965.4, 'fc_om_yr2040': 13330.8, 
                           'fc_om_yr2045': 13620.6, 'fc_om_yr2050': 13910.4 
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 29, new_values_for_pp_id_29.keys()] = new_values_for_pp_id_29.values()

				
##
new_values_for_pp_id_30 = {'cap_pwr_leg_yr2030': 4542,'cap_pwr_leg_yr2035': 13267.5, 'cap_pwr_leg_yr2040': 21993, 
                           'cap_pwr_leg_yr2045': 33496.5, 'cap_pwr_leg_yr2050': 45000, 
                           'fc_om_yr2030': 72500, 'fc_om_yr2035': 60092.5, 'fc_om_yr2040': 47685, 
                           'fc_om_yr2045': 46282.5, 'fc_om_yr2050': 44880 
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 30, new_values_for_pp_id_30.keys()] = new_values_for_pp_id_30.values()

##
new_values_for_pp_id_31 = {'cap_pwr_leg_yr2030': 36531.55,'cap_pwr_leg_yr2035': 48237.15, 'cap_pwr_leg_yr2040': 59942.74, 
                           'cap_pwr_leg_yr2045': 70000.74, 'cap_pwr_leg_yr2050': 70000.74, 
                           'fc_om_yr2030': 72500, 'fc_om_yr2035': 60092.5, 'fc_om_yr2040': 47685, 
                           'fc_om_yr2045': 46282.5, 'fc_om_yr2050': 44880 
                           }				
				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 31, new_values_for_pp_id_31.keys()] = new_values_for_pp_id_31.values()

##
new_values_for_pp_id_32 = { 'cap_pwr_leg_yr2025': 65900.76, 'cap_pwr_leg_yr2030': 215002,
                           'cap_pwr_leg_yr2035': 290438.7, 'cap_pwr_leg_yr2040': 365875.4, 
                           'cap_pwr_leg_yr2045': 400001.02, 'cap_pwr_leg_yr2050': 434126.64, 
                           'fc_om_yr2030': 9500, 'fc_om_yr2035': 9407.5, 'fc_om_yr2040': 9315, 
                           'fc_om_yr2045': 9517.5, 'fc_om_yr2050': 9720 
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 32, new_values_for_pp_id_32.keys()] = new_values_for_pp_id_32.values()

##
new_values_for_pp_id_33 = { 'cap_pwr_leg_yr2025': 3692.03, 'cap_pwr_leg_yr2030': 17369.09,
                           'cap_pwr_leg_yr2035': 28515.01, 'cap_pwr_leg_yr2040': 39660.92, 
                           'cap_pwr_leg_yr2045': 45170.78, 'cap_pwr_leg_yr2050': 50680.64, 
                           'fc_om_yr2030': 9500, 'fc_om_yr2035': 9407.5, 'fc_om_yr2040': 9315, 
                           'fc_om_yr2045': 9517.5, 'fc_om_yr2050': 9720 
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 33, new_values_for_pp_id_33.keys()] = new_values_for_pp_id_33.values()

##
new_values_for_pp_id_34 = {'cap_pwr_leg_yr2030': 74544.27,
                           'cap_pwr_leg_yr2035': 107863.97, 'cap_pwr_leg_yr2040': 141183.67, 
                           'cap_pwr_leg_yr2045': 158451.63, 'cap_pwr_leg_yr2050': 175719.59, 
                           'fc_om_yr2030': 9500, 'fc_om_yr2035': 9407.5, 'fc_om_yr2040': 9315, 
                           'fc_om_yr2045': 9517.5, 'fc_om_yr2050': 9720 
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 34, new_values_for_pp_id_34.keys()] = new_values_for_pp_id_34.values()

##
new_values_for_pp_id_35 = {'cap_pwr_leg_yr2030': 573,
                           'cap_pwr_leg_yr2035': 929.75, 'cap_pwr_leg_yr2040': 1286.5, 
                           'cap_pwr_leg_yr2045': 1643.25, 'cap_pwr_leg_yr2050': 2000, 
                           'fc_om_yr2030': 55764.8, 'fc_om_yr2035': 57381.98, 'fc_om_yr2040': 58999.16, 
                           'fc_om_yr2045': 60281.75, 'fc_om_yr2050': 61564.34 
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 35, new_values_for_pp_id_35.keys()] = new_values_for_pp_id_35.values()

##
new_values_for_pp_id_36 = {'cap_pwr_leg_yr2025': 25600, 'cap_pwr_leg_yr2030': 43441,
                           'cap_pwr_leg_yr2035': 60132.05, 'cap_pwr_leg_yr2040': 76823.1, 
                           'cap_pwr_leg_yr2045': 97411.55, 'cap_pwr_leg_yr2050': 118000, 
                           'fc_om_yr2030': 9500, 'fc_om_yr2035': 9407.5, 'fc_om_yr2040': 9315, 
                           'fc_om_yr2045': 9517.5, 'fc_om_yr2050': 9720 
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 36, new_values_for_pp_id_36.keys()] = new_values_for_pp_id_36.values()


##
new_values_for_pp_id_39 = {'cap_pwr_leg_yr2050': 29690}								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 39, new_values_for_pp_id_39.keys()] = new_values_for_pp_id_39.values()

##
new_values_for_pp_id_52 = {'cap_pwr_leg_yr2030': 168,
                           'cap_pwr_leg_yr2035': 0, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2030': 507451.97, 'erg_chp_yr2035': 0, 'erg_chp_yr2040': 0, 
                           'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 52, new_values_for_pp_id_52.keys()] = new_values_for_pp_id_52.values()


##
new_values_for_pp_id_53 = {'cap_pwr_leg_yr2030': 0,
                           'cap_pwr_leg_yr2035': 0, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2030': 0, 'erg_chp_yr2035': 0, 'erg_chp_yr2040': 0, 
                           'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 53, new_values_for_pp_id_53.keys()] = new_values_for_pp_id_53.values()

##
new_values_for_pp_id_54 = {'cap_pwr_leg_yr2030': 833.38,
                           'cap_pwr_leg_yr2035': 0, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2030': 653683.98, 'erg_chp_yr2035': 0, 'erg_chp_yr2040': 0, 
                           'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 54, new_values_for_pp_id_54.keys()] = new_values_for_pp_id_54.values()

##
new_values_for_pp_id_55 = {'cap_pwr_leg_yr2030': 0,
                           'cap_pwr_leg_yr2035': 0, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2030': 0, 'erg_chp_yr2035': 0, 'erg_chp_yr2040': 0, 
                           'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 55, new_values_for_pp_id_55.keys()] = new_values_for_pp_id_55.values()

##
new_values_for_pp_id_56 = {'cap_pwr_leg_yr2025': 0, 'cap_pwr_leg_yr2030': 0,
                           'cap_pwr_leg_yr2035': 0, 'cap_pwr_leg_yr2040': 0, 
                           'cap_pwr_leg_yr2045': 0, 'cap_pwr_leg_yr2050': 0, 
                           'erg_chp_yr2025': 0,
                           'erg_chp_yr2030': 0, 'erg_chp_yr2035': 0, 'erg_chp_yr2040': 0, 
                           'erg_chp_yr2045': 0, 'erg_chp_yr2050': 0
                           }								
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 56, new_values_for_pp_id_56.keys()] = new_values_for_pp_id_56.values()


##
new_values_for_pp_id_72 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 23500, 'fc_om_yr2050': 23000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 72, new_values_for_pp_id_72.keys()] = new_values_for_pp_id_72.values()
##
new_values_for_pp_id_73 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 23500, 'fc_om_yr2050': 23000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 73, new_values_for_pp_id_73.keys()] = new_values_for_pp_id_73.values()
##
new_values_for_pp_id_74 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 23500, 'fc_om_yr2050': 23000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 74, new_values_for_pp_id_74.keys()] = new_values_for_pp_id_74.values()

##
new_values_for_pp_id_75 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 75, new_values_for_pp_id_75.keys()] = new_values_for_pp_id_75.values()

##
new_values_for_pp_id_76 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 76, new_values_for_pp_id_76.keys()] = new_values_for_pp_id_76.values()

##
new_values_for_pp_id_77 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 77, new_values_for_pp_id_77.keys()] = new_values_for_pp_id_77.values()

##
new_values_for_pp_id_78 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 78, new_values_for_pp_id_78.keys()] = new_values_for_pp_id_78.values()

##
new_values_for_pp_id_79 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 79, new_values_for_pp_id_79.keys()] = new_values_for_pp_id_79.values()

##
new_values_for_pp_id_80 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 80, new_values_for_pp_id_80.keys()] = new_values_for_pp_id_80.values()

##
new_values_for_pp_id_81 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 81, new_values_for_pp_id_81.keys()] = new_values_for_pp_id_81.values()

##
new_values_for_pp_id_82 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 82, new_values_for_pp_id_82.keys()] = new_values_for_pp_id_82.values()

##
new_values_for_pp_id_83 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 83, new_values_for_pp_id_83.keys()] = new_values_for_pp_id_83.values()

##
new_values_for_pp_id_84 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 84, new_values_for_pp_id_84.keys()] = new_values_for_pp_id_84.values()

##
new_values_for_pp_id_85 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 85, new_values_for_pp_id_85.keys()] = new_values_for_pp_id_85.values()

##
new_values_for_pp_id_86 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 86, new_values_for_pp_id_86.keys()] = new_values_for_pp_id_86.values()

##
new_values_for_pp_id_87 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23500, 'fc_om_yr2040': 25000, 
                           'fc_om_yr2045': 28000, 'fc_om_yr2050': 31000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 87, new_values_for_pp_id_87.keys()] = new_values_for_pp_id_87.values()

##
new_values_for_pp_id_88 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23500, 'fc_om_yr2040': 25000, 
                           'fc_om_yr2045': 28000, 'fc_om_yr2050': 31000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 88, new_values_for_pp_id_88.keys()] = new_values_for_pp_id_88.values()

##
new_values_for_pp_id_89 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23500, 'fc_om_yr2040': 25000, 
                           'fc_om_yr2045': 28000, 'fc_om_yr2050': 31000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 89, new_values_for_pp_id_89.keys()] = new_values_for_pp_id_89.values()


##
new_values_for_pp_id_90 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 90, new_values_for_pp_id_90.keys()] = new_values_for_pp_id_90.values()

##
new_values_for_pp_id_91 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 91, new_values_for_pp_id_91.keys()] = new_values_for_pp_id_91.values()

##
new_values_for_pp_id_92 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 92, new_values_for_pp_id_92.keys()] = new_values_for_pp_id_92.values()

##
new_values_for_pp_id_93 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 93, new_values_for_pp_id_93.keys()] = new_values_for_pp_id_93.values()

##
new_values_for_pp_id_94 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 94, new_values_for_pp_id_94.keys()] = new_values_for_pp_id_94.values()

##
new_values_for_pp_id_95 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 95, new_values_for_pp_id_95.keys()] = new_values_for_pp_id_95.values()

##
new_values_for_pp_id_96 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 96, new_values_for_pp_id_96.keys()] = new_values_for_pp_id_96.values()

##
new_values_for_pp_id_97 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 97, new_values_for_pp_id_97.keys()] = new_values_for_pp_id_97.values()

##
new_values_for_pp_id_98 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 98, new_values_for_pp_id_98.keys()] = new_values_for_pp_id_98.values()

##
new_values_for_pp_id_99 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 99, new_values_for_pp_id_99.keys()] = new_values_for_pp_id_99.values()

##
new_values_for_pp_id_100 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 100, new_values_for_pp_id_100.keys()] = new_values_for_pp_id_100.values()

##
new_values_for_pp_id_101 = {'fc_om_yr2030': 22000, 'fc_om_yr2035': 23000, 'fc_om_yr2040': 24000, 
                           'fc_om_yr2045': 26000, 'fc_om_yr2050': 28000
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 101, new_values_for_pp_id_101.keys()] = new_values_for_pp_id_101.values()


##
new_values_for_pp_id_102 = {'fc_om_yr2030': 11000, 'fc_om_yr2035': 11192.5, 'fc_om_yr2040': 11385, 
                           'fc_om_yr2045': 11632.5, 'fc_om_yr2050': 11880
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 102, new_values_for_pp_id_102.keys()] = new_values_for_pp_id_102.values()

##
new_values_for_pp_id_103 = {'fc_om_yr2030': 11000, 'fc_om_yr2035': 11192.5, 'fc_om_yr2040': 11385, 
                           'fc_om_yr2045': 11632.5, 'fc_om_yr2050': 11880
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 103, new_values_for_pp_id_103.keys()] = new_values_for_pp_id_103.values()

##
new_values_for_pp_id_104 = {'fc_om_yr2030': 11000, 'fc_om_yr2035': 11192.5, 'fc_om_yr2040': 11385, 
                           'fc_om_yr2045': 11632.5, 'fc_om_yr2050': 11880
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 104, new_values_for_pp_id_104.keys()] = new_values_for_pp_id_104.values()

##
new_values_for_pp_id_105 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 105, new_values_for_pp_id_105.keys()] = new_values_for_pp_id_105.values()

##
new_values_for_pp_id_106 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 106, new_values_for_pp_id_106.keys()] = new_values_for_pp_id_106.values()

##
new_values_for_pp_id_107 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 107, new_values_for_pp_id_107.keys()] = new_values_for_pp_id_107.values()

##
new_values_for_pp_id_108 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 108, new_values_for_pp_id_108.keys()] = new_values_for_pp_id_108.values()

##
new_values_for_pp_id_109 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 109, new_values_for_pp_id_109.keys()] = new_values_for_pp_id_109.values()

##
new_values_for_pp_id_110 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 110, new_values_for_pp_id_110.keys()] = new_values_for_pp_id_110.values()

##
new_values_for_pp_id_111 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 111, new_values_for_pp_id_111.keys()] = new_values_for_pp_id_111.values()

##
new_values_for_pp_id_112 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 112, new_values_for_pp_id_112.keys()] = new_values_for_pp_id_112.values()

##
new_values_for_pp_id_113 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 113, new_values_for_pp_id_113.keys()] = new_values_for_pp_id_113.values()

##
new_values_for_pp_id_114 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 114, new_values_for_pp_id_114.keys()] = new_values_for_pp_id_114.values()

##
new_values_for_pp_id_115 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 115, new_values_for_pp_id_115.keys()] = new_values_for_pp_id_115.values()

##
new_values_for_pp_id_116 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 116, new_values_for_pp_id_116.keys()] = new_values_for_pp_id_116.values()

##
new_values_for_pp_id_117 = {'fc_om_yr2030': 11000, 'fc_om_yr2035': 11192.5, 'fc_om_yr2040': 11385, 
                           'fc_om_yr2045': 11632.5, 'fc_om_yr2050': 11880
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 117, new_values_for_pp_id_117.keys()] = new_values_for_pp_id_117.values()

##
new_values_for_pp_id_118 = {'fc_om_yr2030': 11000, 'fc_om_yr2035': 11192.5, 'fc_om_yr2040': 11385, 
                           'fc_om_yr2045': 11632.5, 'fc_om_yr2050': 11880
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 118, new_values_for_pp_id_118.keys()] = new_values_for_pp_id_118.values()

##
new_values_for_pp_id_119 = {'fc_om_yr2030': 11000, 'fc_om_yr2035': 11192.5, 'fc_om_yr2040': 11385, 
                           'fc_om_yr2045': 11632.5, 'fc_om_yr2050': 11880
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 119, new_values_for_pp_id_119.keys()] = new_values_for_pp_id_119.values()

##
new_values_for_pp_id_120 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 120, new_values_for_pp_id_120.keys()] = new_values_for_pp_id_120.values()

##
new_values_for_pp_id_121 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 121, new_values_for_pp_id_121.keys()] = new_values_for_pp_id_121.values()

##
new_values_for_pp_id_122 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 122, new_values_for_pp_id_122.keys()] = new_values_for_pp_id_122.values()

##
new_values_for_pp_id_123 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 123, new_values_for_pp_id_123.keys()] = new_values_for_pp_id_123.values()

##
new_values_for_pp_id_124 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 124, new_values_for_pp_id_124.keys()] = new_values_for_pp_id_124.values()

##
new_values_for_pp_id_125 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 125, new_values_for_pp_id_125.keys()] = new_values_for_pp_id_125.values()

##
new_values_for_pp_id_126 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 126, new_values_for_pp_id_126.keys()] = new_values_for_pp_id_126.values()

##
new_values_for_pp_id_127 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 127, new_values_for_pp_id_127.keys()] = new_values_for_pp_id_127.values()

##
new_values_for_pp_id_128 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 128, new_values_for_pp_id_128.keys()] = new_values_for_pp_id_128.values()

##
new_values_for_pp_id_129 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 129, new_values_for_pp_id_129.keys()] = new_values_for_pp_id_129.values()

##
new_values_for_pp_id_130 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 11701.91, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 130, new_values_for_pp_id_130.keys()] = new_values_for_pp_id_130.values()

##
new_values_for_pp_id_131 = {'fc_om_yr2030': 11306.2, 'fc_om_yr2035': 11504.06, 'fc_om_yr2040': 20161.38, 
                           'fc_om_yr2045': 11956.3, 'fc_om_yr2050': 12210.69
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 131, new_values_for_pp_id_131.keys()] = new_values_for_pp_id_131.values()

## VRFB
new_values_for_pp_id_132 = {'fc_om_yr2030': 19479.6, 'fc_om_yr2035': 19820.49, 'fc_om_yr2040': 20161.38, 
                           'fc_om_yr2045': 20599.67, 'fc_om_yr2050': 21037.96
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 132, new_values_for_pp_id_132.keys()] = new_values_for_pp_id_132.values()

##
new_values_for_pp_id_133 = {'fc_om_yr2030': 19479.6, 'fc_om_yr2035': 19820.49, 'fc_om_yr2040': 20161.38, 
                           'fc_om_yr2045': 20599.67, 'fc_om_yr2050': 21037.96
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 133, new_values_for_pp_id_133.keys()] = new_values_for_pp_id_133.values()

##
new_values_for_pp_id_134 = {'fc_om_yr2030': 19479.6, 'fc_om_yr2035': 19820.49, 'fc_om_yr2040': 20161.38, 
                           'fc_om_yr2045': 20599.67, 'fc_om_yr2050': 21037.96
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 134, new_values_for_pp_id_134.keys()] = new_values_for_pp_id_134.values()

##
new_values_for_pp_id_135 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 135, new_values_for_pp_id_135.keys()] = new_values_for_pp_id_135.values()

##
new_values_for_pp_id_136 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 136, new_values_for_pp_id_136.keys()] = new_values_for_pp_id_136.values()

##
new_values_for_pp_id_137 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 137, new_values_for_pp_id_137.keys()] = new_values_for_pp_id_137.values()

##
new_values_for_pp_id_138 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 138, new_values_for_pp_id_138.keys()] = new_values_for_pp_id_138.values()

##
new_values_for_pp_id_139 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 139, new_values_for_pp_id_139.keys()] = new_values_for_pp_id_139.values()

##
new_values_for_pp_id_140 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 140, new_values_for_pp_id_140.keys()] = new_values_for_pp_id_140.values()

##
new_values_for_pp_id_141 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 141, new_values_for_pp_id_141.keys()] = new_values_for_pp_id_141.values()

##
new_values_for_pp_id_142 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 142, new_values_for_pp_id_142.keys()] = new_values_for_pp_id_142.values()

##
new_values_for_pp_id_143 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 143, new_values_for_pp_id_143.keys()] = new_values_for_pp_id_143.values()

##
new_values_for_pp_id_144 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 144, new_values_for_pp_id_144.keys()] = new_values_for_pp_id_144.values()

##
new_values_for_pp_id_145 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 145, new_values_for_pp_id_145.keys()] = new_values_for_pp_id_145.values()

##
new_values_for_pp_id_146 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 146, new_values_for_pp_id_146.keys()] = new_values_for_pp_id_146.values()

##
new_values_for_pp_id_147 = {'fc_om_yr2030': 19479.6, 'fc_om_yr2035': 19820.49, 'fc_om_yr2040': 20161.38, 
                           'fc_om_yr2045': 20599.67, 'fc_om_yr2050': 21037.96
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 147, new_values_for_pp_id_147.keys()] = new_values_for_pp_id_147.values()

##
new_values_for_pp_id_148 = {'fc_om_yr2030': 19479.6, 'fc_om_yr2035': 19820.49, 'fc_om_yr2040': 20161.38, 
                           'fc_om_yr2045': 20599.67, 'fc_om_yr2050': 21037.96
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 148, new_values_for_pp_id_148.keys()] = new_values_for_pp_id_148.values()

##
new_values_for_pp_id_149 = {'fc_om_yr2030': 19479.6, 'fc_om_yr2035': 19820.49, 'fc_om_yr2040': 20161.38, 
                           'fc_om_yr2045': 20599.67, 'fc_om_yr2050': 21037.96
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 149, new_values_for_pp_id_149.keys()] = new_values_for_pp_id_149.values()

##
new_values_for_pp_id_150 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 150, new_values_for_pp_id_150.keys()] = new_values_for_pp_id_150.values()

##
new_values_for_pp_id_151 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 151, new_values_for_pp_id_151.keys()] = new_values_for_pp_id_151.values()

##
new_values_for_pp_id_152 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 152, new_values_for_pp_id_152.keys()] = new_values_for_pp_id_152.values()

##
new_values_for_pp_id_153 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 153, new_values_for_pp_id_153.keys()] = new_values_for_pp_id_153.values()

##
new_values_for_pp_id_154 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 154, new_values_for_pp_id_154.keys()] = new_values_for_pp_id_154.values()

##
new_values_for_pp_id_155 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 155, new_values_for_pp_id_155.keys()] = new_values_for_pp_id_155.values()

##
new_values_for_pp_id_156 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 156, new_values_for_pp_id_156.keys()] = new_values_for_pp_id_156.values()

##
new_values_for_pp_id_157 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 157, new_values_for_pp_id_157.keys()] = new_values_for_pp_id_157.values()

##
new_values_for_pp_id_158 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 158, new_values_for_pp_id_158.keys()] = new_values_for_pp_id_158.values()

##
new_values_for_pp_id_159 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 159, new_values_for_pp_id_159.keys()] = new_values_for_pp_id_159.values()

##
new_values_for_pp_id_160 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 160, new_values_for_pp_id_160.keys()] = new_values_for_pp_id_160.values()

##
new_values_for_pp_id_161 = {'fc_om_yr2030': 24556.18, 'fc_om_yr2035': 24985.92, 'fc_om_yr2040': 25415.65, 
                           'fc_om_yr2045': 25968.16, 'fc_om_yr2050': 26520.68
                           }				
df_plant_encar_0.loc[df_plant_encar_0['pp_id'] == 161, new_values_for_pp_id_161.keys()] = new_values_for_pp_id_161.values()

df_plant_encar_0.to_csv(data_path + '\\plant_encar.csv', index=False)
#%%
# Secondly, here we add the new power plants in plant_encar
# (as a reference we rely on def_plant - and consider profsupply for Wind Offshore/def_pp_type for Storage)

dict_pp_new_plants = pd.Series(df_def_plant.pp_id.values,index=df_def_plant.pp).to_dict()
# dict_nd_id_all = dict(pd.Series(df_def_node_0.nd_id.values,index=df_def_node_0.nd).to_dict(), **dict_nd_id)
dict_pt_id_all = dict(pd.Series(df_def_pp_type_0.pt_id.values,index=df_def_pp_type_0.pt).to_dict(),
                      **pd.Series(df_def_pp_type.pt_id.values,index=df_def_pp_type.pt))

df_plant_encar_new_plants = pd.read_csv(base_dir + '\\update_data_with_recent_scenarios\\plant_encar_new_plants.csv')

df_plant_encar_new_plants.rename(columns={'pp': 'pp_id'}, inplace=True)

df_plant_encar_new_plants.loc[df_plant_encar_new_plants['pp_id'] == 'IT_WIN_OFF', 'supply_pf_id'] =  dict_pf_id['supply_IT_WIN_OFF']


df_plant_encar_new_plants['pp_id'] = df_plant_encar_new_plants['pp_id'].map(dict_pp_new_plants)


df_plant_encar_0_new_plants = pd.concat([df_plant_encar_new_plants, df_plant_encar_0])

df_plant_encar_0_new_plants.to_csv(data_path + '\\plant_encar.csv', index=False)

df_plant_encar_new_plants

# %% ~~~~~~~~~~~~~~~~~~~~ NODE_CONNECT EV
# --> NO CHANGES!

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FUEL_NODE_ENCAR
# 

df_fuel_node_encar_0 = pd.read_csv(data_path_prv + '/fuel_node_encar.csv')


# Define the list of fl_id values you want to update
fl_ids_to_update = [0]
new_values_for_fl_id = {
    'vc_fl_yr2030': 6.41, 'vc_fl_yr2035': 6.08, 'vc_fl_yr2040': 5.76, 'vc_fl_yr2045': 5.62, 'vc_fl_yr2050': 5.47
                        }
# Iterate over the list of fl_id values and update each one
for fl_id in fl_ids_to_update:
    df_fuel_node_encar_0.loc[df_fuel_node_encar_0['fl_id'] == fl_id, new_values_for_fl_id.keys()] = new_values_for_fl_id.values()

# Define the list of fl_id values you want to update
fl_ids_to_update = [1]
new_values_for_fl_id = {
    'vc_fl_yr2030': 6.48, 'vc_fl_yr2035': 6.48, 'vc_fl_yr2040': 6.48, 'vc_fl_yr2045': 6.48, 'vc_fl_yr2050': 6.48
                        }
for fl_id in fl_ids_to_update:
    df_fuel_node_encar_0.loc[df_fuel_node_encar_0['fl_id'] == fl_id, new_values_for_fl_id.keys()] = new_values_for_fl_id.values()

# Define the list of fl_id values you want to update
fl_ids_to_update = [2]
new_values_for_fl_id = {
    'vc_fl_yr2030': 18.63, 'vc_fl_yr2035': 17.84, 'vc_fl_yr2040': 17.05, 'vc_fl_yr2045': 16.27, 'vc_fl_yr2050': 15.48
                        }
for fl_id in fl_ids_to_update:
    df_fuel_node_encar_0.loc[df_fuel_node_encar_0['fl_id'] == fl_id, new_values_for_fl_id.keys()] = new_values_for_fl_id.values()

# Define the list of fl_id values you want to update
fl_ids_to_update = [3]
new_values_for_fl_id = {
    'vc_fl_yr2030': 6.05, 'vc_fl_yr2035': 6.05, 'vc_fl_yr2040': 6.05, 'vc_fl_yr2045': 6.05, 'vc_fl_yr2050': 6.05
                        }
for fl_id in fl_ids_to_update:
    df_fuel_node_encar_0.loc[df_fuel_node_encar_0['fl_id'] == fl_id, new_values_for_fl_id.keys()] = new_values_for_fl_id.values()

# Define the list of fl_id values you want to update
fl_ids_to_update = [13]
new_values_for_fl_id = {
    'vc_fl_yr2030': 38.19, 'vc_fl_yr2035': 36.38, 'vc_fl_yr2040': 34.57, 'vc_fl_yr2045': 32.77, 'vc_fl_yr2050': 30.96
                        }
for fl_id in fl_ids_to_update:
    df_fuel_node_encar_0.loc[df_fuel_node_encar_0['fl_id'] == fl_id, new_values_for_fl_id.keys()] = new_values_for_fl_id.values()



### Here we apply changes for erg_inp!!!

# Define a dictionary with (fl_id, nd_id) as keys and dictionaries of new values as values
updates_to_apply = {
    (0, 0): {
        'erg_inp_yr2025': 0,
        'erg_inp_yr2030': 0,
        'erg_inp_yr2035': 0,
        'erg_inp_yr2040': 0,
        'erg_inp_yr2045': 0,
        'erg_inp_yr2050': 0
    },
    (0, 4): {
        'erg_inp_yr2030': 1200000,
        'erg_inp_yr2035': 1200000,
        'erg_inp_yr2040': 0,
        'erg_inp_yr2045': 0,
        'erg_inp_yr2050': 0
    },
    (2, 0): { 				
        'erg_inp_yr2030': 11514450,
        'erg_inp_yr2035': 9926100,
        'erg_inp_yr2040': 8337750,
        'erg_inp_yr2045': 8321120,
        'erg_inp_yr2050': 8304490
    },
    (2, 1): { 				
        'erg_inp_yr2030': 11514450,
        'erg_inp_yr2035': 9926100,
        'erg_inp_yr2040': 8337750,
        'erg_inp_yr2045': 8321120,
        'erg_inp_yr2050': 8304490
    },    				
    (2, 2): { 				
        'erg_inp_yr2030': 50819000,
        'erg_inp_yr2035': 34000000,
        'erg_inp_yr2040': 34000000,
        'erg_inp_yr2045': 23000000,
        'erg_inp_yr2050': 23000000
    },
    (2, 3): { 								
        'erg_inp_yr2030': 19182500,
        'erg_inp_yr2035': 12009395.95,
        'erg_inp_yr2040': 4836291.9,
        'erg_inp_yr2045': 2637805.5,
        'erg_inp_yr2050': 439319.1
    },
    (2, 4): { 								
        'erg_inp_yr2030': 75000000,
        'erg_inp_yr2035': 62500000,
        'erg_inp_yr2040': 50000000,
        'erg_inp_yr2045': 37500000,
        'erg_inp_yr2050': 25000000
    },
    (3, 3): { 								
        'erg_inp_yr2040':295197035.2,		
        'erg_inp_yr2045': 270121018,
        'erg_inp_yr2050': 181500553.6
    },
    (8, 0): { 								
        'erg_inp_yr2030': 21132325.34,
        'erg_inp_yr2035': 34693145.28,
        'erg_inp_yr2040': 48253965.22,
        'erg_inp_yr2045': 54957606.81,
        'erg_inp_yr2050': 61661248.4
    },
    (8, 1): { 								
        'erg_inp_yr2030': 772678.0654,
        'erg_inp_yr2035': 1253747.699,
        'erg_inp_yr2040': 1734817.332,
        'erg_inp_yr2045': 2215886.965,
        'erg_inp_yr2050': 2696956.598
    },
    (8, 2): { 								
        'erg_inp_yr2030': 236717202,
        'erg_inp_yr2035': 319773008.7,
        'erg_inp_yr2040': 402828815.4,
        'erg_inp_yr2045': 440401123,
        'erg_inp_yr2050': 477973430.6
    },
    (8, 3): { 								
        'erg_inp_yr2030': 53208715.22,
        'erg_inp_yr2035': 73652750.26,
        'erg_inp_yr2040': 94096785.3,
        'erg_inp_yr2045': 119314551.3,
        'erg_inp_yr2050': 144532317.3
    },
    (8, 4): { 								
        'erg_inp_yr2030': 100770125.1,
        'erg_inp_yr2035': 145812229.1,
        'erg_inp_yr2040': 190854333,
        'erg_inp_yr2045': 214197436.3,
        'erg_inp_yr2050': 237540539.5
    },
    (21, 0): { 								
        'erg_inp_yr2030': 17864029.63,
        'erg_inp_yr2035': 21266701.94,
        'erg_inp_yr2040': 24669374.25,
        'erg_inp_yr2045': 30398810.96,
        'erg_inp_yr2050': 36128247.66
    },
    (21, 2): { 								
        'erg_inp_yr2030': 295189404.1,
        'erg_inp_yr2035': 365329050.4,
        'erg_inp_yr2040': 435468696.7,
        'erg_inp_yr2045': 466674407.2,
        'erg_inp_yr2050': 468519920.7
    },
    (21, 3): { 								
        'erg_inp_yr2030': 93182416.71,
        'erg_inp_yr2035': 139240339.4,
        'erg_inp_yr2040': 185298262.1,
        'erg_inp_yr2045': 242952082.1,
        'erg_inp_yr2050': 300605902.1
    },
    (21, 4): { 								
        'erg_inp_yr2030': 58354916.12,
        'erg_inp_yr2035': 73609366.11,
        'erg_inp_yr2040': 88863816.09,
        'erg_inp_yr2045': 99508695.71,
        'erg_inp_yr2050': 110153575.3
    },    

}

# Iterate through the updates_to_apply dictionary
for (fl_id, nd_id), new_values in updates_to_apply.items():
    # Apply updates for each (fl_id, nd_id) pair using the corresponding new values
    df_fuel_node_encar_0.loc[(df_fuel_node_encar_0['fl_id'] == fl_id) & 
                             (df_fuel_node_encar_0['nd_id'] == nd_id), 
                             new_values.keys()] = new_values.values()


df_fuel_node_encar_0.to_csv(data_path + '\\fuel_node_encar.csv', index=False)
#
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DF_PROFPRICE
#
# --> NO CHANGES!

# %%





list_tb_app = { #'def_node': df_def_node, 
               # 'def_encar':df_def_encar,
                'def_pp_type': df_def_pp_type,
                # 'def_fuel': df_def_fuel,
                # 'node_connect': df_node_connect,
                'def_plant': df_def_plant,
                'def_profile': df_def_profile,
                # 'plant_encar' : df_plant_encar_new_plants,
                # 'node_encar' : df_node_encar_add,
                'profsupply' : df_profsupply,
                # 'profdmnd' : df_profdmnd_add, 

                }
list_tb_new = {                  
    }


import glob
import pathlib
csv_files_previous = glob.glob(os.path.join(data_path_prv, "*.csv"))

for f in csv_files_previous:
      
    # read the csv file
    df_prv = pd.read_csv(f)
    table_name = f.split("/")[-1][:-4]
    path = pathlib.PurePath(f)
    
    if path.name[:-4] in list_tb_app.keys():
        df_app = pd.concat([df_prv,list_tb_app[path.name[:-4]]])
        df_app.to_csv(os.path.join(data_path, '%s.csv'%path.name[:-4]), index=False)
        print('Table append to previous data:',f.split("/")[-1])
    else:
        pass
    

#%%
#import glob
#csv_files_previous = glob.glob(os.path.join(data_path, "*.csv"))

#for f in csv_files_previous:
      
    # read the csv file
 #   df_prv = pd.read_csv(f)
  #  table_name = f.split("/")[-1][:-4]
    
   # if table_name in list_tb_app.keys():
    #    df_app = pd.concat([df_prv,list_tb_app[table_name]])
     #   df_app.to_csv(os.path.join(data_path, '%s.csv'%table_name), index=False)
      #  print('Table append to previous data:',f.split("/")[-1])
    #elif table_name in list_tb_new.keys():
     #   df_new = list_tb_new[table_name]
      #  df_new.to_csv(os.path.join(data_path, '%s.csv'%table_name), index=False)
       # print('Table new replacing previous data:',f.split("/")[-1])

   # else:
    #    df_prv.to_csv(os.path.join(data_path, '%s.csv'%table_name), index=False)
     #   print('Tabel no change compare to previous data:',f.split("/")[-1])












