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
data_path = conf.PATH_CSV + '\\csv_files_ev'
data_path_prv = conf.PATH_CSV + '\\csv_files_dsr_ee_dhw'
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


#%% EV loads

# dfload_arch_ev = pd.read_csv(base_dir+ '/ev/demand/dmnd_archetypes_ev.csv',sep=';')
dfload_arch_ev = pd.read_csv(base_dir+ '/ev/demand/dmnd_archetypes_ev.csv')
dfload_arch_ev['DateTime'] = dfload_arch_ev['DateTime'].astype('datetime64[ns]')

dferg_arch_ev = dfload_arch_ev.groupby('nd_id')['erg_tot'].sum().reset_index()
dferg_arch_ev['nd_id_new'] = dferg_arch_ev.nd_id

dfload_arch_ev['nd_id_new'] = dfload_arch_ev.nd_id

# %% ~~~~~~~~~~~~~~~~~~   DEF_NODE (we add EV nodes)

color_nd = {            
            'SFH_URB_0_EV':       '#818789',
            'SFH_URB_1_EV':       '#818789',
            'SFH_URB_2_EV':       '#818789',
            'SFH_URB_3_EV':       '#818789',
            'SFH_SUB_0_EV':       '#6D3904',
            'SFH_SUB_1_EV':       '#6D3904',
            'SFH_SUB_2_EV':       '#6D3904',
            'SFH_SUB_3_EV':       '#6D3904',
            'SFH_RUR_0_EV':       '#0A81EE',
            'SFH_RUR_1_EV':       '#0A81EE',
            'SFH_RUR_2_EV':       '#0A81EE',
            'SFH_RUR_3_EV':       '#0A81EE',           
            'MFH_URB_0_EV':       '#484A4B',
            'MFH_URB_1_EV':       '#484A4B',
            'MFH_URB_2_EV':       '#484A4B',
            'MFH_URB_3_EV':       '#484A4B',
            'MFH_SUB_0_EV':       '#041FA3',
            'MFH_SUB_1_EV':       '#041FA3',
            'MFH_SUB_2_EV':       '#041FA3',
            'MFH_SUB_3_EV':       '#041FA3',
            'MFH_RUR_0_EV':       '#472503',
            'MFH_RUR_1_EV':       '#472503',
            'MFH_RUR_2_EV':       '#472503',
            'MFH_RUR_3_EV':       '#472503',

            }


col_nd_df = pd.DataFrame.from_dict(color_nd, orient='index').reset_index().rename(columns={'index': 'nd',0:'color'})

df_def_node_0 = pd.read_csv(data_path_prv + '\\def_node.csv')

df_nd_add = pd.DataFrame(dferg_arch_ev.nd_id_new.unique()).rename(columns={0:'nd'}
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

# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PP_TYPE

# ---> NO CHANGES

# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_FUEL for EVs

# ---> NO CHANGES

# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_ENCAR for EV

#  ----> NO CHANGES

# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PLANT

#  ----> NO CHANGES
#

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEF_PROFILE for EV

df_def_profile_0 = pd.read_csv(data_path_prv + '\\def_profile.csv')

# Remeber we have 2015 demand
# Demand profile for EV

df_def_profile_EV = df_nd_add.nd.copy().rename('primary_nd').reset_index()
df_def_profile_EV ['pf'] = 'demand_EL_' + df_def_profile_EV.primary_nd
df_def_profile_EV ['pf_id'] = df_def_profile_EV.index.rename('pf_id') + df_def_profile_0.pf_id.max() + 1
df_def_profile_EV = df_def_profile_EV[df_def_profile_0.columns]


df_def_profile = pd.concat([df_def_profile_EV,], axis=0)
df_def_profile = df_def_profile.reset_index(drop=True)
    
df_def_profile

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NODE_ENCAR for EV

df_node_encar_0 = pd.read_csv(data_path_prv + '/node_encar.csv')

df_ndca_add_EV = (dferg_arch_ev.loc[dfload_arch_ev.nd_id_new.isin(df_def_node.nd), ['nd_id_new', 'erg_tot']]
                          .rename(columns={'erg_tot': 'dmnd_sum', 'nd_id_new': 'nd_id'}))

data_0 = dict(vc_dmnd_flex=0, ca_id=0, grid_losses=0, grid_losses_absolute=0)

df_node_encar_EV_0 = df_ndca_add_EV.assign(**data_0).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
df_node_encar_EV_0 = pd.merge(df_node_encar_EV_0, df_def_profile_EV, left_on='nd_id', right_on='primary_nd', how='inner')

list_dmnd = [c for c in df_node_encar_EV_0 if 'dmnd_sum' in c]
df_node_encar_EV_0 = df_node_encar_EV_0.assign(**{c: df_node_encar_EV_0.dmnd_sum
                                        for c in list_dmnd})



fct_dmnd = pd.read_csv(base_dir+'/ev/demand/factor_dmnd_EV100%_future_years_2050.csv',sep=';')

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

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFDMND for EV

df_tm_st =pd.read_csv(base_dir+'/ev/timemap/timestamp_template.csv')

df_tm_st['datetime'] = df_tm_st['datetime'].astype('datetime64[ns]')

df_profdmnd_0 = pd.read_csv(data_path_prv + '/profdmnd.csv').head()

df_dmnd_ev_add = dfload_arch_ev.copy()

df_dmnd_ev_add  = pd.merge(dfload_arch_ev, df_def_profile_EV[['pf_id', 'primary_nd']], left_on='nd_id_new', right_on='primary_nd')

df_dmnd_ev_add = df_dmnd_ev_add.rename(columns={'erg_tot': 'value', 'pf_id': 'dmnd_pf_id'})
df_dmnd_ev_add['doy'] = (df_dmnd_ev_add.hy + 24)//24

df_dmnd_ev_add['erg_tot_fossil'] = 0
df_dmnd_ev_add['erg_tot_retr_1pc'] = 0
df_dmnd_ev_add['erg_tot_retr_2pc'] = 0
df_dmnd_ev_add['erg_tot_retr_1pc_fossil'] = 0
df_dmnd_ev_add['erg_tot_retr_2pc_fossil'] = 0

df_profdmnd_add = df_dmnd_ev_add[df_profdmnd_0.columns.tolist()].reset_index(drop=True)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFPRICE

# --> NO CHANGES! 
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFSUPPLY

# --> NO CHANGES!
# %% ~~~~~~~~~~~~~~~~~~~~~~~ PLANT_ENCAR (needs profsupply data)

# --> NO CHANGES!

# %% ~~~~~~~~~~~~~~~~~~~~ NODE_CONNECT EV

df_node_connect_0 = pd.read_csv(data_path_prv + '/node_connect.csv').head()


node_res_el = df_nd_res_el.nd.values
node_res_ev = df_def_node.loc[df_def_node.nd.str.contains('EV')].nd.values
data_res = dict(nd_id=node_res_el, nd_2_id=node_res_ev, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0)
data_res_df = pd.DataFrame(data_res)
data_res_df = expand_rows(data_res_df, ['mt_id'], [range(12)])
data_res_df['mt_id'] = data_res_df.mt_id.astype(int)
df_node_connect_ev = data_res_df[df_node_connect_0.columns]

dft = pd.concat([pd.read_csv(data_path_prv + '/def_node.csv'), df_def_node])
#
for idx in [('nd'), (['nd', 'nd_2'])]:

    df_node_connect, _ = translate_id(df_node_connect_ev, dft, idx)
    
df_node_connect = df_node_connect.reset_index(drop=True)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FUEL_NODE_ENCAR
# --> NO CHANGES!

#
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DF_PROFPRICE
#
# --> NO CHANGES!

# %%





list_tb_app = {'def_node': df_def_node, 
               # 'def_encar':df_def_encar,
                  # 'def_pp_type':df_def_pp_type,
                # 'def_fuel': df_def_fuel,
                'node_connect': df_node_connect,
                # 'def_plant': df_def_plant,
                'def_profile': df_def_profile,
                # 'plant_encar' : df_plant_encar,
                'node_encar' : df_node_encar_add,
                'profdmnd' : df_profdmnd_add, 

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












