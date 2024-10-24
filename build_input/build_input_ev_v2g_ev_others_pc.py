# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:35:09 2019

@author: user
"""

# execute primary input data building script
#import build_input.build_input_ev
print('####################')
print('BUILDING INPUT DATA FOR EV CHARGING OTHER PLACES AND PUBLIC CHARGING')
print('####################')
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

data_path = conf.PATH_CSV + '\\csv_files_ev_res_ev_others_v2g'
data_path_prv = conf.PATH_CSV + '\\csv_files_ev_v2g'
base_dir = conf.BASE_DIR

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
    
#%%

# We read old files for EV consumption (first paper) - build_inputs_files_ev.

#First paper EV demand/profiles
old_ev = pd.read_csv(base_dir + '\\ev\\demand\\dmnd_archetypes_ev.csv')

#Second paper EV profiles + additional nodes
new_ev = pd.read_csv(base_dir + '\\ev_v2g_ev_others_pc\\demand\\dmnd_archetypes_ev_res_ev_others_v2g.csv')
new_ev_sfh_mfh = new_ev[new_ev['nd_id'].str.contains('SFH|MFH')]
#To substitute previous values with the new one (existing EV node), we read files from data_path_prv.

def_profile_prv = pd.read_csv(data_path_prv + '\\def_profile.csv')
def_profile_prv = def_profile_prv[def_profile_prv.primary_nd.str.contains('EV')]
def_profile_prv_list = def_profile_prv[def_profile_prv.primary_nd.str.contains('EV')].pf_id.tolist()
EV_list_prv = def_profile_prv[def_profile_prv.primary_nd.str.contains('EV')].pf_id.tolist()
old_ev_dmnd = pd.read_csv(data_path_prv + '\\profdmnd.csv')
old_ev_dmnd_ev = old_ev_dmnd[old_ev_dmnd.dmnd_pf_id.isin(EV_list_prv)]

EV_old_dmnd_pf_id = old_ev_dmnd_ev.dmnd_pf_id.unique().tolist()

ev_old_dmnd_pf_id_set = set(EV_old_dmnd_pf_id)

# Select rows where dmnd_pf_id is not in EV_old_dmnd_pf_id
filtered_df = old_ev_dmnd[~old_ev_dmnd['dmnd_pf_id'].isin(ev_old_dmnd_pf_id_set)]

# Step 1: Merge def_profile_prv with new_ev to map nd_id to pf_id
merged_df = pd.merge(new_ev_sfh_mfh, def_profile_prv[['pf_id', 'primary_nd']], left_on='nd_id', right_on='primary_nd')

# Step 2: Rename 'pf_id' to 'dmnd_pf_id' for the merge with old_ev_dmnd
merged_df = merged_df.rename(columns={'pf_id': 'dmnd_pf_id', 'erg_tot': 'value'})

merged_df['doy'] = (merged_df.hy + 24)//24
merged_df['erg_tot_fossil'] = 0
merged_df['erg_tot_retr_1pc'] = 0
merged_df['erg_tot_retr_2pc'] = 0
merged_df['erg_tot_retr_1pc_fossil'] = 0
merged_df['erg_tot_retr_2pc_fossil'] = 0

# Reorder the columns of merged_df to match the order in filtered_df
merged_df_reordered = merged_df[filtered_df.columns]
final_df = pd.concat([filtered_df, merged_df_reordered], ignore_index=True)

final_df.to_csv(data_path + '\\profdmnd.csv',  index=False)

df_def_prof_demand = pd.read_csv(data_path + '\\profdmnd.csv')
#%% 

df_node_encar_old = pd.read_csv(data_path_prv + '\\node_encar.csv')
df_def_profile_old_ev = pd.read_csv(data_path_prv + '\\def_profile.csv')

list_old_ev = df_def_profile_old_ev[df_def_profile_old_ev.pf.str.contains('EV')].pf_id.tolist()
list_old_ev_nd = df_def_profile_old_ev[df_def_profile_old_ev.pf.str.contains('EV')]

df_node_encar_old_ev = df_node_encar_old[df_node_encar_old.dmnd_pf_id.isin(list_old_ev)]

dmnd_pf_id_to_nd_id_dict = df_node_encar_old_ev.set_index('dmnd_pf_id')['nd_id'].to_dict()

df_node_encar_old_ev_fl = df_node_encar_old_ev[df_node_encar_old_ev.dmnd_pf_id.isin(list_old_ev)]
summed_values = merged_df.groupby(by=['dmnd_pf_id', 'nd_id'])['value'].sum().rename('summed_value').reset_index() 
summed_value_series = summed_values.set_index('dmnd_pf_id')['summed_value']
df_node_encar_old_ev_fl['dmnd_sum'] = df_node_encar_old_ev_fl['dmnd_pf_id'].map(summed_value_series)
df_node_encar_old_ev_fl['dmnd_sum_yr2020'] = df_node_encar_old_ev_fl['dmnd_sum']
df_node_encar_old_ev_fl['dmnd_sum_yr2025'] = df_node_encar_old_ev_fl['dmnd_sum']
df_node_encar_old_ev_fl['dmnd_sum_yr2030'] = df_node_encar_old_ev_fl['dmnd_sum']
df_node_encar_old_ev_fl['dmnd_sum_yr2035'] = df_node_encar_old_ev_fl['dmnd_sum']
df_node_encar_old_ev_fl['dmnd_sum_yr2040'] = df_node_encar_old_ev_fl['dmnd_sum']
df_node_encar_old_ev_fl['dmnd_sum_yr2045'] = df_node_encar_old_ev_fl['dmnd_sum']
df_node_encar_old_ev_fl['dmnd_sum_yr2050'] = df_node_encar_old_ev_fl['dmnd_sum']

pf_id_mapping = list_old_ev_nd.set_index('primary_nd')['pf_id']

fct_dmnd_old_ev = pd.read_csv(base_dir+'/ev/demand/factor_dmnd_EV100%_future_years_2050.csv',sep=';')
fct_dmnd_old_ev['nd_id'] = fct_dmnd_old_ev['nd_id'].map(pf_id_mapping)
fct_dmnd_old_ev['nd_id'] = fct_dmnd_old_ev['nd_id'].map(dmnd_pf_id_to_nd_id_dict)

df_0 = df_node_encar_old_ev_fl.set_index(['nd_id','ca_id']).filter(like='dmnd_sum')
fct_dmnd_old_ev = fct_dmnd_old_ev.set_index(['nd_id','ca_id']).filter(like='dmnd_sum')
df_0 = df_0*fct_dmnd_old_ev
df_node_encar_old_ev_fl = df_node_encar_old_ev_fl.set_index(['nd_id','ca_id'])
df_node_encar_old_ev_fl.update(df_0)
df_node_encar_old_ev_fl = df_node_encar_old_ev_fl.reset_index()

df_node_encar_old = df_node_encar_old[:-24] 
df_node_encar_old_ev_fl

final_node_encar_old = pd.concat([df_node_encar_old, df_node_encar_old_ev_fl], ignore_index=True)
final_node_encar_old.to_csv(data_path + '\\node_encar.csv',  index=False)

df_node_encar_old_no_ev = pd.read_csv(data_path + '\\node_encar.csv')


#%% Load files for EV charging profiles at IND/OCO + PC (Public Charging) 

# EV IND/OCO plus PC
dfload_EV_ind_oco = pd.read_csv(base_dir + '\\ev_v2g_ev_others_pc\\demand\\dmnd_archetypes_ev_res_ev_others_v2g.csv')
# dfload_EV_ind_oco['DateTime'] = pd.to_datetime(dfload_EV_ind_oco.DateTime)

dfload_EV_ind_oco = dfload_EV_ind_oco[~dfload_EV_ind_oco['nd_id'].str.contains('SFH|MFH')]
dfload_EV_ind_oco = dfload_EV_ind_oco.reset_index(drop=True)
dfload_EV_ind_oco['index'] = dfload_EV_ind_oco.index

dfload_EV_ind_oco['nd_id_new'] = dfload_EV_ind_oco.nd_id

dferg_arch_ev_ind_oco = dfload_EV_ind_oco.groupby('nd_id')['erg_tot'].sum().reset_index()
dferg_arch_ev_ind_oco['nd_id_new'] = dferg_arch_ev_ind_oco.nd_id


#%%
# Def_node for EV -> IND/OCO + PC (Public Charging) -> Add nodes


color_nd = {            
            'IND_RUR_EV':       '#818789',
            'IND_SUB_EV':       '#818789',
            'IND_URB_EV':       '#818789',
            'OCO_RUR_EV':       '#818789',
            'OCO_SUB_EV':       '#6D3904',
            'OCO_URB_EV':       '#6D3904',            
            'Public_Charging':       '#6D3904',
            
            }


col_nd_df = pd.DataFrame.from_dict(color_nd, orient='index').reset_index().rename(columns={'index': 'nd',0:'color'})

df_def_node_0 = pd.read_csv(data_path_prv + '\\def_node.csv')

df_nd_add = pd.DataFrame(dfload_EV_ind_oco.nd_id_new.unique()).rename(columns={0:'nd'}
                                                                    ).reset_index(drop=True)

#
nd_id_max = df_def_node_0.loc[~df_def_node_0.nd.isin(df_nd_add.nd)].nd_id.max()
df_nd_add['nd_id'] = np.arange(0, len(df_nd_add)) + nd_id_max + 1
#
df_nd_add = pd.merge(df_nd_add,col_nd_df, on = 'nd')
#
df_def_node = df_def_node_0.copy()
df_def_node = df_nd_add.reindex(columns=df_def_node_0.columns.tolist()).fillna(0).reset_index(drop=True)
#
dict_nd_id_ev = df_nd_add.set_index('nd')['nd_id'].to_dict()
#
df_nd_res_el = df_def_node_0.loc[~df_def_node_0.nd.str.contains('HT|DSR|EV') & df_def_node_0.nd.str.contains('SFH|MFH')]
df_nd_not_res = df_def_node_0.loc[~df_def_node_0.nd.str.contains('MFH|SFH')]
dict_nd_res_el = df_nd_res_el.set_index('nd')['nd_id'].to_dict()

# %% ~~~~~~~~~~~~~~~~~~~~ DEF_PP_TYPE for additonal EV nodes (IND_EV, OCO_EV, and Public_Charging)

# ---> NO CHANGES
# %% ~~~~~~~~~~~~~~~~~~~~ DEF_FUEL for additonal EV nodes (IND_EV, OCO_EV, and Public_Charging)

# ---> NO CHANGES
# %% ~~~~~~~~~~~~~~~~~~~~ DEF_ENCAR for additonal EV nodes (IND_EV, OCO_EV, and Public_Charging)

# ---> NO CHANGES
# %% ~~~~~~~~~~~~~~~~~~~~ DEF_PLANT for additonal EV nodes (IND_EV, OCO_EV, and Public_Charging)

# ---> NO CHANGES
# %% ~~~~~~~~~~~~~~~~~~~~ DEF_PROFILE for additonal EV nodes (IND_EV, OCO_EV, and Public_Charging)


df_def_profile_0 = pd.read_csv(data_path_prv + '\\def_profile.csv')

# Remeber we have 2015 demand
# Demand profile for EV -> (IND_EV, OCO_EV, and Public_Charging)

df_def_profile_EV = df_nd_add.nd.copy().rename('primary_nd').reset_index()
df_def_profile_EV ['pf'] = 'demand_EL_' + df_def_profile_EV.primary_nd
df_def_profile_EV ['pf_id'] = df_def_profile_EV.index.rename('pf_id') + df_def_profile_0.pf_id.max() + 1
df_def_profile_EV = df_def_profile_EV[df_def_profile_0.columns]


df_def_profile = pd.concat([df_def_profile_EV,], axis=0)
df_def_profile = df_def_profile.reset_index(drop=True)
    
df_def_profile
# %% ~~~~~~~~~~~~~~~~~~~~ NODE_ENCAR for additonal EV nodes (IND_EV, OCO_EV, and Public_Charging)

# df_node_encar_0 = pd.read_csv(data_path_prv + '/node_encar.csv')
df_node_encar_0 = pd.read_csv(data_path + '/node_encar.csv')

df_ndca_add_EV = (dferg_arch_ev_ind_oco.loc[dfload_EV_ind_oco.nd_id_new.isin(df_def_node.nd), ['nd_id_new', 'erg_tot']]
                          .rename(columns={'erg_tot': 'dmnd_sum', 'nd_id_new': 'nd_id'}))

data_0 = dict(vc_dmnd_flex=0, ca_id=0, grid_losses=0, grid_losses_absolute=0)

df_node_encar_EV_0 = df_ndca_add_EV.assign(**data_0).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
df_node_encar_EV_0 = pd.merge(df_node_encar_EV_0, df_def_profile_EV, left_on='nd_id', right_on='primary_nd', how='inner')

list_dmnd = [c for c in df_node_encar_EV_0 if 'dmnd_sum' in c]
df_node_encar_EV_0 = df_node_encar_EV_0.assign(**{c: df_node_encar_EV_0.dmnd_sum
                                        for c in list_dmnd})



fct_dmnd = pd.read_csv(base_dir+'/ev_v2g_ev_others_pc/demand/factor_dmnd_EV100%_future_years_2050.csv',sep=';')

df_0 = df_node_encar_EV_0.set_index(['nd_id','ca_id']).filter(like='dmnd_sum')
fct_dmnd = fct_dmnd.set_index(['nd_id','ca_id']).filter(like='dmnd_sum')
df_0 = df_0*fct_dmnd
df_node_encar_EV = df_node_encar_EV_0.set_index(['nd_id','ca_id'])
df_node_encar_EV.update(df_0)
df_node_encar_EV = df_node_encar_EV.reset_index()


df_node_encar_EV['dmnd_pf_id'] = df_node_encar_EV.pf
df_node_encar_EV = df_node_encar_EV.loc[:, df_node_encar_0.columns]

for df, idx in [(pd.concat([df_def_node_0,df_def_node]), 'nd'), (df_def_profile, ['pf', 'dmnd_pf'])]:
    df_node_encar_EV, _ = translate_id(df_node_encar_EV, df, idx)

df_node_encar_add = pd.concat([df_node_encar_EV,
                               ])
df_node_encar_add = df_node_encar_add.reset_index(drop=True)
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFDMND for additonal EV nodes (IND_EV, OCO_EV, and Public_Charging)


df_tm_st = pd.read_csv(base_dir+'/ev_v2g_ev_others_pc/timemap/timestamp_template.csv')

df_tm_st['datetime'] = df_tm_st['datetime'].astype('datetime64[ns]')

df_profdmnd_0 = pd.read_csv(data_path_prv + '/profdmnd.csv').head()

df_dmnd_ev_others_pc_add = dfload_EV_ind_oco.copy()

df_dmnd_ev_others_pc_add  = pd.merge(dfload_EV_ind_oco, df_def_profile_EV[['pf_id', 'primary_nd']], left_on='nd_id_new', right_on='primary_nd')

df_dmnd_ev_others_pc_add = df_dmnd_ev_others_pc_add.rename(columns={'erg_tot': 'value', 'pf_id': 'dmnd_pf_id'})
df_dmnd_ev_others_pc_add['doy'] = (df_dmnd_ev_others_pc_add.hy + 24)//24

df_dmnd_ev_others_pc_add['erg_tot_fossil'] = 0
df_dmnd_ev_others_pc_add['erg_tot_retr_1pc'] = 0
df_dmnd_ev_others_pc_add['erg_tot_retr_2pc'] = 0
df_dmnd_ev_others_pc_add['erg_tot_retr_1pc_fossil'] = 0
df_dmnd_ev_others_pc_add['erg_tot_retr_2pc_fossil'] = 0

df_profdmnd_add = df_dmnd_ev_others_pc_add[df_profdmnd_0.columns.tolist()].reset_index(drop=True)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFPRICE

# --> NO CHANGES! HOUSEHOLDS USE CH0 PRICE PROFILES
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFSUPPLY

# --> NO CHANGES!
# %% ~~~~~~~~~~~~~~~~~~~~~~~ PLANT_ENCAR for others - the capacities will be changed in mlm script

# %% ~~~~~~~~~~~~~~~~~~~~ NODE_CONNECT for new added nodes (IND, OCO, and PC)

df_node_connect_0 = pd.read_csv(data_path_prv + '/node_connect.csv').head()

node_ind_oco_pc_el = df_nd_add.nd.values

node_oco_rur = df_def_node_0.loc[df_def_node_0.nd.str.contains('OCO_RUR')].nd.values
node_oco_sub = df_def_node_0.loc[df_def_node_0.nd.str.contains('OCO_SUB')].nd.values
node_oco_urb = df_def_node_0.loc[df_def_node_0.nd.str.contains('OCO_URB')].nd.values
node_ind_rur = df_def_node_0.loc[df_def_node_0.nd.str.contains('IND_RUR')].nd.values
node_ind_sub = df_def_node_0.loc[df_def_node_0.nd.str.contains('IND_SUB')].nd.values
node_ind_urb = df_def_node_0.loc[df_def_node_0.nd.str.contains('IND_URB')].nd.values

node_oco_rur_ev = df_def_node.loc[df_def_node.nd.str.contains('OCO_RUR')].nd.values
node_oco_sub_ev = df_def_node.loc[df_def_node.nd.str.contains('OCO_SUB')].nd.values
node_oco_urb_ev = df_def_node.loc[df_def_node.nd.str.contains('OCO_URB')].nd.values
node_ind_rur_ev = df_def_node.loc[df_def_node.nd.str.contains('IND_RUR')].nd.values
node_ind_sub_ev = df_def_node.loc[df_def_node.nd.str.contains('IND_SUB')].nd.values
node_ind_urb_ev = df_def_node.loc[df_def_node.nd.str.contains('IND_URB')].nd.values

node_ch0 = df_def_node_0.loc[df_def_node_0.nd.str.contains('CH0')].nd.values
node_public_charging = df_def_node.loc[df_def_node.nd.str.contains('Public')].nd.values

# Connection EV for OCO and IND
data_oco_rur = dict(nd_id=node_oco_rur, nd_2_id=node_oco_rur_ev, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_oco_sub = dict(nd_id=node_oco_sub, nd_2_id=node_oco_sub_ev, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_oco_urb = dict(nd_id=node_oco_urb, nd_2_id=node_oco_urb_ev, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)

data_ind_rur = dict(nd_id=node_ind_rur, nd_2_id=node_ind_rur_ev, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_ind_sub = dict(nd_id=node_ind_sub, nd_2_id=node_ind_sub_ev, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_ind_urb = dict(nd_id=node_ind_urb, nd_2_id=node_ind_urb_ev, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)

# Connection Public_Charging for CH node
data_ch0 = dict(nd_id=node_ch0, nd_2_id=node_public_charging, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)


# data_ind_df = pd.concat([pd.DataFrame(data_ind_rur),pd.DataFrame(data_ind_sub),pd.DataFrame(data_ind_urb)])
# data_ind_df = expand_rows(data_ind_df, ['mt_id'], [range(12)])
# data_ind_df[['mt_id']] = data_ind_df.mt_id.astype(int)
# df_node_connect = data_ind_df[df_node_connect_0.columns]

data_ind_oco_pc_df = pd.concat([
                    pd.DataFrame(data_oco_rur),pd.DataFrame(data_oco_sub),pd.DataFrame(data_oco_urb),
                    pd.DataFrame(data_ind_rur),pd.DataFrame(data_ind_sub),pd.DataFrame(data_ind_urb),
                    pd.DataFrame(data_ch0)
                             ])

data_ind_oco_pc_df = expand_rows(data_ind_oco_pc_df, ['mt_id'], [range(12)])
data_ind_oco_pc_df[['mt_id']] = data_ind_oco_pc_df.mt_id.astype(int)
df_node_connect = data_ind_oco_pc_df[df_node_connect_0.columns]


dft = pd.concat([df_def_node_0, df_def_node])
#
for idx in [('nd'), (['nd', 'nd_2'])]:

    df_node_connect, _ = translate_id(df_node_connect, dft, idx)
    
df_node_connect = df_node_connect.reset_index(drop=True)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FUEL_NODE_ENCAR
# --> NO CHANGES!

#
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DF_PROFPRICE
#
##df_profprice = aql.read_sql(db, sc, 'profprice') <- NO CHANGE

#%%  Update the following files to add the changes for V2G in csv - input_data files! 


list_tb_app = { 'def_node': df_def_node, 
               # 'def_encar':df_def_encar,
                   # 'def_pp_type':df_def_pp_type,
                # 'def_fuel': df_def_fuel,
                'node_connect': df_node_connect,
                 # 'def_plant': df_def_plant,
                'def_profile': df_def_profile,
                 # 'plant_encar' : df_plant_encar_new,
                'node_encar' : df_node_encar_add,
                'profdmnd' : df_profdmnd_add, 

                }
list_tb_new = {                  
    }


import glob
import pathlib
csv_files_previous = glob.glob(os.path.join(data_path, "*.csv"))

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
    

# %%

# #list_tb_col = [
# #           (df_plant_encar_new, 'plant_encar', ['pp_id', 'ca_id'])
# #           ]

# list_tb_new = [
# #            (df_def_plant_new, 'def_plant', ['pp_id']),
#            # (df_node_encar_new, 'node_encar', ['nd_id', 'ca_id']),
#            # (df_profdmnd_new, 'profdmnd', ['dmnd_pf_id']),
#            (df_plant_encar_new, 'plant_encar', ['pp_id', 'ca_id']),
#            ]

# list_tb = [
# #        (df_fuel_node_encar, 'fuel_node_encar', ['nd_id', 'fl_id', 'ca_id']),
#           (df_node_encar, 'node_encar', ['nd_id', 'ca_id']),
#           (df_profdmnd_add, 'profdmnd', ['dmnd_pf_id']),
# #           (df_profsupply, 'profsupply', ['supply_pf_id']), <-- NO CHANGE
# #           (df_profprice, 'profprice', ['price_pf_id']),, <-- NO CHANGE
#           (df_node_connect, 'node_connect', ['nd_id', 'nd_2_id']),
# #           (df_plant_encar_new, 'plant_encar', ['pp_id', 'ca_id']),
#            (df_def_plant, 'def_plant', ['pp_id']),
#            (df_def_node, 'def_node', ['nd_id']),
#            (df_def_encar, 'def_encar', ['ca_id']),
#            (df_def_fuel, 'def_fuel', ['fl_id']), 
#            (df_def_pp_type, 'def_pp_type', ['pt_id']),
#            (df_def_profile, 'def_profile', ['pf_id'])
#            ]

# # tables with foreign keys first
# #df, tb, ind = (df_def_plant, 'def_plant', ['pp_id'])

# #df, tb = (df_def_plant_new, 'def_plant')
# #replace_table(df,tb)
# # df, tb = (df_def_node, 'def_node')
# # replace_table(df,tb)
# df, tb = (df_plant_encar_new, 'plant_encar')
# #append_new_cols(df, tb)
# replace_table(df,tb)
# # df, tb = (df_node_encar_new, 'node_encar')
# # replace_table(df,tb)
# # df, tb = (df_profdmnd_new, 'profdmnd')
# # replace_table(df,tb)
# #for df, tb, ind in list_tb_col:
# #    print('Replacing table %s'%tb)
# #    append_new_cols(df, tb)


# for df, tb, ind in list_tb:
#     print('Deleting from table %s'%tb)
#     del_new_rows(ind, tb, df)

# # tables with foreign keys last
# for df, tb, ind in reversed(list_tb):
#     print('Appending to table %s'%tb)
#     append_new_rows(df, tb)





# for tb in aql.get_sql_tables(sc, db):
#     print(tb)
#     df = aql.read_sql(db, sc, tb)

#     if 'prof' in tb and 'value' in df.columns:
#         df['value'] = df['value'].round(13)

#     df.to_csv(os.path.join(data_path, '%s.csv'%tb), index=False)