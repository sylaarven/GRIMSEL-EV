# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:48 2024

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
data_path = conf.PATH_CSV + '\\csv_files_ev_v2g'
data_path_prv = conf.PATH_CSV + '\\csv_files_new_ev'
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


#%% V2G loads

# dfload_arch_ev = pd.read_csv(base_dir+ '/ev/demand/dmnd_archetypes_ev.csv',sep=';')
dfload_arch_v2g = pd.read_csv(base_dir+ '/v2g/v2g/v2g_plant_encar.csv')

# dfload_arch_v2g['pp_new'] = dfload_arch_v2g.pp

# dfload_arch_v2g['nd_id_new'] = dfload_arch_v2g.nd_id

# %% ~~~~~~~~~~~~~~~~~~~~~~~ Def_node for V2G 

# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PP_TYPE for V2G

df_def_pp_type_0 = pd.read_csv(data_path_prv + '\\def_pp_type.csv')

df_def_pp_type = df_def_pp_type_0.copy().head(0)

for npt, pt, cat, color in ((0, 'V2G_SFH', 'NEW_STORAGE_V2G_SFH', '#00ff00'),
                            (1, 'V2G_MFH', 'NEW_STORAGE_V2G_MFH', '#ff00ff'),
                            ):

    df_def_pp_type.loc[npt] = (npt, pt, cat, color)


df_def_pp_type.loc[:,'pt_id'] = np.arange(0, len(df_def_pp_type)) + df_def_pp_type_0.pt_id.max() + 1




# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_FUEL for V2G
# df_def_fuel_0 = pd.read_csv(data_path_prv + '/def_fuel.csv')

# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_ENCAR for V2G


# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PLANT for V2G


df_def_plant_0 = pd.read_csv(data_path_prv + '/def_plant.csv')
df_def_node_0 = pd.read_csv(data_path_prv + '/def_node.csv')
df_def_fuel_0 = pd.read_csv(data_path_prv + '/def_fuel.csv')


df_nd_res_el = df_def_node_0.loc[df_def_node_0.nd.str.contains('MFH|SFH')]
df_nd_not_res = df_def_node_0.loc[~df_def_node_0.nd.str.contains('MFH|SFH')]
df_nd_arch_el = df_def_node_0.loc[(df_def_node_0.nd.str.contains('_0|_1|_2|_3')) & ~df_def_node_0.nd.str.contains('DSR') & ~df_def_node_0.nd.str.contains('EV') & ~df_def_node_0.nd.str.contains('HT')]
df_nd_ch0_el = df_def_node_0.loc[df_def_node_0.nd.str.contains('CH0')]


df_pp_add_arch = pd.DataFrame(df_nd_arch_el.nd).rename(columns={'nd': 'nd_id'})


# df_def_plant_v2g = pd.read_csv(data_path + '/def_plant.csv')

# df_pp_add_h2 = pd.DataFrame(df_nd_add.nd).rename(columns={'nd': 'nd_id'})
# df_pp_add_h2 = pd.DataFrame(df_nd_ind_el.nd).rename(columns={'nd': 'nd_id'})
#df_pp_add_v2g = pd.DataFrame(pd.concat([df_nd_sfh_el,df_nd_mfh_el]).nd).rename(columns={'nd': 'nd_id'})


# Adding first V2G storage (arch el level)
df_pp_add_1 = df_pp_add_arch.nd_id.str.slice(stop=9)
df_pp_add = pd.DataFrame()

for sfx, fl_id, pt_id, set_1 in [('_V2G', 'new_storage', 'STO_V2G_', ['set_def_st',]),
                                 ]:

    new_pp_id = df_def_plant_0.pp_id.max() + 1
    data = dict(pp=df_pp_add_arch + sfx,
                fl_id=fl_id, pt_id=pt_id + df_pp_add_1 , pp_id=np.arange(new_pp_id, new_pp_id + len(df_pp_add_arch)),
                **{st: 1 if st in set_1 else 0 for st in [c for c in df_def_plant_0.columns if 'set' in c]})

    df_pp_add = df_pp_add.append(df_pp_add_arch.assign(**data), sort=True)

df_pp_add.pp_id = np.arange(0, len(df_pp_add)) + df_pp_add.pp_id.min()


df_def_plant = df_pp_add[df_def_plant_0.columns].reset_index(drop=True)

for df, idx in [(pd.concat([df_def_fuel_0]), 'fl'), (df_def_pp_type, 'pt'), (pd.concat([df_def_node_0]), 'nd')]:

    df_def_plant, _ = translate_id(df_def_plant, df, idx)
    
# Create a dictionary from df_def_pp_type dataframe
replace_dict_pt_id = df_def_pp_type.set_index('pt')['pt_id'].to_dict()
replace_dict_nd_id = df_nd_arch_el.set_index('nd')['nd_id'].to_dict()

# Now replace the 'pt_id' values in df_def_plant with dictionary values
df_def_plant['pt_id'] = df_def_plant['pt_id'].replace(replace_dict_pt_id, regex=True)
# df_def_plant['nd_id'] = df_def_plant['nd_id'].replace(replace_dict_nd_id, regex=True)

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEF_PROFILE for V2G

# --> NO CHANGES!
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NODE_ENCAR for V2G

# --> NO CHANGES!
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFDMND for V2G

# --> NO CHANGES!
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFPRICE

# --> NO CHANGES! HOUSEHOLDS USE CH0 PRICE PROFILES
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFSUPPLY

# --> NO CHANGES!
# %% ~~~~~~~~~~~~~~~~~~~~~~~ PLANT_ENCAR for V2G - the capacities will be changed in mlm script

df_plant_encar_0 = pd.read_csv(data_path_prv + '\\plant_encar.csv')
dict_pp_new_v2g = pd.Series(df_def_plant.pp_id.values,index=df_def_plant.pp).to_dict()
# dict_nd_id_all = dict(pd.Series(df_def_node_0.nd_id.values,index=df_def_node_0.nd).to_dict(), **dict_nd_id)
dict_pt_id_all = dict(pd.Series(df_def_pp_type_0.pt_id.values,index=df_def_pp_type_0.pt).to_dict(),
                      **pd.Series(df_def_pp_type.pt_id.values,index=df_def_pp_type.pt))


dict_pt_id_v2g = dict(pd.Series(df_def_pp_type.pt_id.values,index=df_def_pp_type.pt).to_dict(),
                      **pd.Series(df_def_pp_type.pt_id.values,index=df_def_pp_type.pt))

df_plant_encar_v2g = pd.read_csv(base_dir + '\\v2g\\v2g\\v2g_plant_encar.csv')

df_plant_encar_v2g.rename(columns={'pp': 'pp_id'}, inplace=True)


df_nd_id = df_nd_arch_el.set_index('nd')['nd_id'].to_dict()
# df_plant_encar_v2g['pt_id'] = df_plant_encar_v2g['pt_id'].replace(dict_pt_id_v2g, regex=True)
df_plant_encar_v2g['pp_id'] = df_plant_encar_v2g['pp_id'].replace(dict_pp_new_v2g, regex=True)

# df_plant_encar_v2g = df_plant_encar_v2g.reset_index().rename(columns={'pp_id': 'index'})
df_plant_encar_1 = pd.DataFrame(columns=df_plant_encar_0.columns)
df_plant_encar_2 = pd.concat([df_plant_encar_1, df_plant_encar_v2g], ignore_index=True)


df_plant_encar_new = df_plant_encar_2
# %% ~~~~~~~~~~~~~~~~~~~~ NODE_CONNECT

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FUEL_NODE_ENCAR
# --> NO CHANGES!
#
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DF_PROFPRICE
#
##df_profprice = aql.read_sql(db, sc, 'profprice') <- NO CHANGE

#%%  Update the following files to add the changes for V2G in csv - input_data files! 

  
    
list_tb_app = { #'def_node': df_def_node, 
               # 'def_encar':df_def_encar,
                   'def_pp_type':df_def_pp_type,
                # 'def_fuel': df_def_fuel,
                # 'node_connect': df_node_connect,
                 'def_plant': df_def_plant,
                #'def_profile': df_def_profile,
                 'plant_encar' : df_plant_encar_new,
                #'node_encar' : df_node_encar_add,
                #'profdmnd' : df_profdmnd_add, 

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


#%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# %%

#list_tb_col = [
#           (df_plant_encar_new, 'plant_encar', ['pp_id', 'ca_id'])
#           ]

list_tb_new = [
#            (df_def_plant_new, 'def_plant', ['pp_id']),
           # (df_node_encar_new, 'node_encar', ['nd_id', 'ca_id']),
           # (df_profdmnd_new, 'profdmnd', ['dmnd_pf_id']),
           (df_plant_encar_new, 'plant_encar', ['pp_id', 'ca_id']),
           ]

list_tb = [
#        (df_fuel_node_encar, 'fuel_node_encar', ['nd_id', 'fl_id', 'ca_id']),
          (df_node_encar, 'node_encar', ['nd_id', 'ca_id']),
          (df_profdmnd_add, 'profdmnd', ['dmnd_pf_id']),
#           (df_profsupply, 'profsupply', ['supply_pf_id']), <-- NO CHANGE
#           (df_profprice, 'profprice', ['price_pf_id']),, <-- NO CHANGE
          (df_node_connect, 'node_connect', ['nd_id', 'nd_2_id']),
#           (df_plant_encar_new, 'plant_encar', ['pp_id', 'ca_id']),
           (df_def_plant, 'def_plant', ['pp_id']),
           (df_def_node, 'def_node', ['nd_id']),
           (df_def_encar, 'def_encar', ['ca_id']),
           (df_def_fuel, 'def_fuel', ['fl_id']), 
           (df_def_pp_type, 'def_pp_type', ['pt_id']),
           (df_def_profile, 'def_profile', ['pf_id'])
           ]

# tables with foreign keys first
#df, tb, ind = (df_def_plant, 'def_plant', ['pp_id'])

#df, tb = (df_def_plant_new, 'def_plant')
#replace_table(df,tb)
# df, tb = (df_def_node, 'def_node')
# replace_table(df,tb)
df, tb = (df_plant_encar_new, 'plant_encar')
#append_new_cols(df, tb)
replace_table(df,tb)
# df, tb = (df_node_encar_new, 'node_encar')
# replace_table(df,tb)
# df, tb = (df_profdmnd_new, 'profdmnd')
# replace_table(df,tb)
#for df, tb, ind in list_tb_col:
#    print('Replacing table %s'%tb)
#    append_new_cols(df, tb)


for df, tb, ind in list_tb:
    print('Deleting from table %s'%tb)
    del_new_rows(ind, tb, df)

# tables with foreign keys last
for df, tb, ind in reversed(list_tb):
    print('Appending to table %s'%tb)
    append_new_rows(df, tb)





for tb in aql.get_sql_tables(sc, db):
    print(tb)
    df = aql.read_sql(db, sc, tb)

    if 'prof' in tb and 'value' in df.columns:
        df['value'] = df['value'].round(13)

    df.to_csv(os.path.join(data_path, '%s.csv'%tb), index=False)




