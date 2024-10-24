# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:35:09 2019

@author: user
"""

# execute primary input data building script
import build_input.build_input_ev
print('####################')
print('BUILDING INPUT DATA FOR INCLUDING HYDROGEN AND TRANSPORT')
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

base_dir = conf.BASE_DIR
data_path = conf.PATH_CSV
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

#%% H2 demand

# H2 industry
dfload_ind_h2 = pd.read_csv(base_dir + '/h2_transport/demand/dmnd_h2_ind_hhv.csv', sep=';')
dfload_ind_h2['DateTime'] = pd.to_datetime(dfload_ind_h2.DateTime)


dferg_ind_h2 = dfload_ind_h2.groupby('nd_id')['erg_tot'].sum().reset_index()
dferg_ind_h2['nd_id_new'] = dferg_ind_h2.nd_id

dfload_ind_h2['erg_tot'] = dfload_ind_h2.erg_tot/24 # MWh -> MW

# H2 transport
dfload_transp_h2 = pd.read_csv(base_dir + '/h2_transport/demand/dmnd_h2_transp_hhv.csv', sep=';')
dfload_transp_h2['DateTime'] = pd.to_datetime(dfload_transp_h2.DateTime)


dferg_transp_h2 = dfload_transp_h2.groupby('nd_id')['erg_tot'].sum().reset_index()
dferg_transp_h2['nd_id_new'] = dferg_transp_h2.nd_id

dfload_transp_h2['erg_tot'] = dfload_transp_h2.erg_tot/24 # MWh -> MW


# EL transport LCV HCV
dfload_transp_el = pd.read_csv(base_dir + '/h2_transport/demand/dmnd_el_transp_daily.csv', sep=';')
dfload_transp_el['DateTime'] = pd.to_datetime(dfload_transp_el.DateTime)


dferg_transp_el = dfload_transp_el.groupby('nd_id')['erg_tot'].sum().reset_index()
dferg_transp_el['nd_id_new'] = dferg_transp_el.nd_id

dfload_transp_el['erg_tot'] = dfload_transp_el.erg_tot/24 # MWh -> MW


# All
# dferg_h2 = pd.concat([dferg_ind_h2,dferg_transp_h2],ignore_index=True)
dferg_h2_el = pd.concat([dferg_ind_h2,dferg_transp_h2,dferg_transp_el],ignore_index=True)



color_nd = {'IND_H2':       '#0000ff',
            'TRANSP_H2':       '#0000ff',
            'TRANSP_EL':       '#0000ff',
            }

col_nd_df = pd.DataFrame.from_dict(color_nd, orient='index').reset_index().rename(columns={'index': 'nd',0:'color'})

df_def_node_0 = pd.read_csv(data_path + '/def_node.csv')

# df_nd_add = pd.DataFrame(dferg_ind_h2.nd_id_new.unique()).rename(columns={0:'nd'}
#                                                                    ).reset_index(drop=True)
# df_nd_add = pd.DataFrame(dferg_h2.nd_id_new.unique()).rename(columns={0:'nd'}
#                                                                    ).reset_index(drop=True)
df_nd_add = pd.DataFrame(dferg_h2_el.nd_id_new.unique()).rename(columns={0:'nd'}
                                                                   ).reset_index(drop=True)

                         
#
nd_id_max = df_def_node_0.loc[~df_def_node_0.nd.isin(df_nd_add.nd)].nd_id.max()
df_nd_add['nd_id'] = np.arange(0, len(df_nd_add)) + nd_id_max + 1
#
df_nd_add = pd.merge(df_nd_add,col_nd_df, on = 'nd')
#
df_def_node = df_def_node_0.copy()
df_def_node = df_nd_add.reindex(columns=df_def_node_0.columns.tolist()).fillna(0).reset_index(drop=True)

# dict_nd_id_dsr = df_nd_add.set_index('nd')['nd_id'].to_dict()
# dict_nd_id_ind_h2 = df_nd_add.set_index('nd')['nd_id'].to_dict()
dict_nd_id_h2 = df_nd_add.set_index('nd')['nd_id'].to_dict()
dict_nd_id_ind_h2 = df_nd_add.loc[df_nd_add.nd.str.contains('IND')].set_index('nd')['nd_id'].to_dict()
dict_nd_id_transp_h2 = df_nd_add.loc[df_nd_add.nd.str.contains('TRANS_H2')].set_index('nd')['nd_id'].to_dict()
dict_nd_id_transp_el = df_nd_add.loc[df_nd_add.nd.str.contains('TRANS_EL')].set_index('nd')['nd_id'].to_dict()

# df_nd_res_el = df_def_node_0.loc[~df_def_node_0.nd.str.contains('HT') & df_def_node_0.nd.str.contains('SFH|MFH')]
# df_nd_not_res = df_def_node_0.loc[~df_def_node_0.nd.str.contains('MFH|SFH')]
# df_nd_arch_el = df_def_node_0.loc[~df_def_node_0.nd.str.contains('HT') & df_def_node_0.nd.str.contains('SFH|MFH|OCO|IND')]
# df_nd_arch_ht = df_def_node_0.loc[df_def_node_0.nd.str.contains('HT')]
df_nd_ch0_el = df_def_node_0.loc[df_def_node_0.nd.str.contains('CH0')]
df_nd_ind_el = df_def_node_0.loc[df_def_node_0.nd.str.contains('IND')]
df_nd_oco_el = df_def_node_0.loc[df_def_node_0.nd.str.contains('OCO')]


# dict_nd_res_el = df_nd_res_el.set_index('nd')['nd_id'].to_dict()
# dict_nd_arch_ht = df_nd_arch_ht.set_index('nd')['nd_id'].to_dict()
dict_nd_id_ind_el = df_nd_ind_el.set_index('nd')['nd_id'].to_dict()
dict_nd_id_ind_ch0_el = pd.concat([df_nd_ind_el,df_nd_ch0_el]).set_index('nd')['nd_id'].to_dict()
dict_nd_id_oco_el = df_nd_oco_el.set_index('nd')['nd_id'].to_dict()

# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PP_TYPE

df_def_pp_type_0 = pd.read_csv(data_path + '/def_pp_type.csv')

df_def_pp_type = df_def_pp_type_0.copy().head(0)

for npt, pt, cat, color in ((0, 'H2_ELY_IND', 'H2_ELECTROLYZER_IND', '#00ff00'),
                            (1, 'H2_STO_IND', 'H2_STORAGE_IND', '#ff00ff'),
                            (2, 'H2_FC_IND', 'H2_FUELCELL_IND', '#ff8000'),
                            (3, 'H2_ELY_CH0', 'H2_ELECTROLYZER_CH0', '#00ff00'),
                            (4, 'H2_STO_CH0', 'H2_STORAGE_CH0', '#ff00ff'),
                            (5, 'H2_FC_CH0', 'H2_FUELCELL_CH0', '#ff8000'),
                            ):

    df_def_pp_type.loc[npt] = (npt, pt, cat, color)


df_def_pp_type.loc[:,'pt_id'] = np.arange(0, len(df_def_pp_type)) + df_def_pp_type_0.pt_id.max() + 1




# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_FUEL for DHW

# 
df_def_fuel_0 = pd.read_csv(data_path + '/def_fuel.csv')
# df_def_fuel_0 = aql.read_sql(db, sc, 'def_fuel')

df_def_fuel = df_def_fuel_0.copy().head(0)

for nfl, fl, co2_int, ca, constr, color in ((0, 'ca_h2', 0,0,0, 'b'),
                                           (1, 'h2_storage', 0,0,0, 'r'),
                                            ):
#                

    df_def_fuel.loc[nfl] = (nfl, fl, co2_int, ca, constr, color)


df_def_fuel.loc[:,'fl_id'] = np.arange(0, len(df_def_fuel)) + df_def_fuel_0.fl_id.max() + 1


# %% ~~~~~~~~~~~~~~~~~~~~~~ DEF_ENCAR for DHW

# 
df_def_encar_0 = pd.read_csv(data_path + '/def_encar.csv')
# df_def_encar_0 = aql.read_sql(db, sc, 'def_encar')

df_def_encar = df_def_encar_0.copy().head(0)

for nca, fl_id, ca in ((0, 31, 'H2'),
                       ):
    
    df_def_encar.loc[nca] = (nca, fl_id, ca)

df_def_encar.loc[:,'ca_id'] = np.arange(0, len(df_def_encar)) + df_def_encar_0.ca_id.max() + 1


# %% ~~~~~~~~~~~~~~~~~~~~~~~ DEF_PLANT


df_def_plant_0 = pd.read_csv(data_path + '/def_plant.csv')
# df_def_plant_0 = aql.read_sql(db, sc, 'def_plant')



# df_pp_add_h2 = pd.DataFrame(df_nd_add.nd).rename(columns={'nd': 'nd_id'})
# df_pp_add_h2 = pd.DataFrame(df_nd_ind_el.nd).rename(columns={'nd': 'nd_id'})
df_pp_add_h2 = pd.DataFrame(pd.concat([df_nd_ind_el,df_nd_ch0_el]).nd).rename(columns={'nd': 'nd_id'})

# df_pp_add_arch = pd.DataFrame(df_nd_res_el.nd).rename(columns={'nd': 'nd_id'})


# df_pp_add_1 = df_pp_add_arch.nd_id.str.slice(stop=3)
df_pp_add_1 = df_pp_add_h2.nd_id.str.slice(stop=3)
df_pp_add = pd.DataFrame()


for sfx, fl_id, pt_id, set_1 in [('_H2_ELY', 'ca_electricity', 'H2_ELY_', ['set_def_pp','set_def_add']),
                                 ('_H2_STO', 'h2_storage', 'H2_STO_', ['set_def_st','set_def_add']),
                                 ('_H2_FC', 'ca_h2', 'H2_FC_', ['set_def_pp','set_def_add']),
                                 ]:
    
    new_pp_id = df_def_plant_0.pp_id.max() + 1
    data = dict(pp=df_pp_add_h2 + sfx,
                fl_id=fl_id, pt_id=pt_id + df_pp_add_1, pp_id=np.arange(new_pp_id, new_pp_id + len(df_pp_add_h2)),
                **{st: 1 if st in set_1 else 0 for st in [c for c in df_def_plant_0.columns if 'set' in c]})

    df_pp_add = df_pp_add.append(df_pp_add_h2.assign(**data), sort=True)

df_pp_add.pp_id = np.arange(0, len(df_pp_add)) + df_pp_add.pp_id.min()


df_def_plant = df_pp_add[df_def_plant_0.columns].reset_index(drop=True)

for df, idx in [(pd.concat([df_def_fuel_0,df_def_fuel]), 'fl'), (df_def_pp_type, 'pt'), (pd.concat([df_def_node_0]), 'nd')]:

    df_def_plant, _ = translate_id(df_def_plant, df, idx)

#df_def_plant_new = pd.concat([df_def_plant_0,df_def_plant]).reset_index(drop=True)
# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEF_PROFILE for H2 and transport

df_def_profile_0 = pd.read_csv(data_path + '/def_profile.csv')

# Industry H2 demand profiles

# df_def_profile_dmnd_ind_h2 = df_nd_add.nd.copy().rename('primary_nd').reset_index()
# df_def_profile_dmnd_ind_h2['pf'] = 'demand_H2_' + df_def_profile_dmnd_ind_h2.primary_nd
# df_def_profile_dmnd_ind_h2['pf_id'] = df_def_profile_dmnd_ind_h2.index.rename('pf_id') + df_def_profile_0.pf_id.max() + 1
# df_def_profile_dmnd_ind_h2 = df_def_profile_dmnd_ind_h2[df_def_profile_0.columns]

# # Demand el profiles in H2 nodes
# # df_def_profile_dmnd_ind_h2_el = df_def_node.nd.copy().rename('primary_nd').reset_index()
# # df_def_profile_dmnd_ind_h2_el['pf'] = 'demand_EL_' + df_def_profile_dmnd_ind_h2_el.primary_nd
# # df_def_profile_dmnd_ind_h2_el['pf_id'] = df_def_profile_dmnd_ind_h2_el.index.rename('pf_id') + df_def_profile_dmnd_ind_h2.pf_id.max() + 1
# # df_def_profile_dmnd_ind_h2_el = df_def_profile_dmnd_ind_h2_el[df_def_profile_0.columns]

# # Demand H2 profiles in ind nodes
# df_def_profile_dmnd_ind_h2_el = df_nd_ind_el.nd.copy().rename('primary_nd').reset_index()
# df_def_profile_dmnd_ind_h2_el['pf'] = 'demand_H2_' + df_def_profile_dmnd_ind_h2_el.primary_nd
# df_def_profile_dmnd_ind_h2_el['pf_id'] = df_def_profile_dmnd_ind_h2_el.index.rename('pf_id') + df_def_profile_dmnd_ind_h2.pf_id.max() + 1
# df_def_profile_dmnd_ind_h2_el = df_def_profile_dmnd_ind_h2_el[df_def_profile_0.columns]

# # Industry and Transport H2 demand profiles

# df_def_profile_dmnd_ind_transp_h2 = df_nd_add.nd.copy().rename('primary_nd').reset_index()
# df_def_profile_dmnd_ind_transp_h2['pf'] = 'demand_H2_' + df_def_profile_dmnd_ind_transp_h2.primary_nd
# df_def_profile_dmnd_ind_transp_h2['pf_id'] = df_def_profile_dmnd_ind_transp_h2.index.rename('pf_id') + df_def_profile_0.pf_id.max() + 1
# df_def_profile_dmnd_ind_transp_h2 = df_def_profile_dmnd_ind_transp_h2[df_def_profile_0.columns]

# # Demand el profiles in H2 nodes
# # df_def_profile_dmnd_ind_h2_el = df_def_node.nd.copy().rename('primary_nd').reset_index()
# # df_def_profile_dmnd_ind_h2_el['pf'] = 'demand_EL_' + df_def_profile_dmnd_ind_h2_el.primary_nd
# # df_def_profile_dmnd_ind_h2_el['pf_id'] = df_def_profile_dmnd_ind_h2_el.index.rename('pf_id') + df_def_profile_dmnd_ind_h2.pf_id.max() + 1
# # df_def_profile_dmnd_ind_h2_el = df_def_profile_dmnd_ind_h2_el[df_def_profile_0.columns]

# # Demand H2 profiles in IND and  CH0 node (for the coupling)
# df_def_profile_dmnd_ind_ch0_h2_el = pd.concat([df_nd_ind_el,df_nd_ch0_el]).nd.copy().rename('primary_nd').reset_index()
# df_def_profile_dmnd_ind_ch0_h2_el['pf'] = 'demand_H2_' + df_def_profile_dmnd_ind_ch0_h2_el.primary_nd
# df_def_profile_dmnd_ind_ch0_h2_el['pf_id'] = df_def_profile_dmnd_ind_ch0_h2_el.index.rename('pf_id') + df_def_profile_dmnd_ind_transp_h2.pf_id.max() + 1
# df_def_profile_dmnd_ind_ch0_h2_el= df_def_profile_dmnd_ind_ch0_h2_el[df_def_profile_0.columns]

# Industry and Transport H2 and Transport EL demand profiles

# Industry and Transport H2
df_def_profile_dmnd_ind_transp_h2 = df_nd_add.loc[df_nd_add.nd.str.contains('H2')].nd.copy().rename('primary_nd').reset_index()
df_def_profile_dmnd_ind_transp_h2['pf'] = 'demand_H2_' + df_def_profile_dmnd_ind_transp_h2.primary_nd
df_def_profile_dmnd_ind_transp_h2['pf_id'] = df_def_profile_dmnd_ind_transp_h2.index.rename('pf_id') + df_def_profile_0.pf_id.max() + 1
df_def_profile_dmnd_ind_transp_h2 = df_def_profile_dmnd_ind_transp_h2[df_def_profile_0.columns]

# Transport EL
df_def_profile_dmnd_ind_transp_el = df_nd_add.loc[df_nd_add.nd.str.contains('EL')].nd.copy().rename('primary_nd').reset_index()
df_def_profile_dmnd_ind_transp_el['pf'] = 'demand_EL_' + df_def_profile_dmnd_ind_transp_el.primary_nd
df_def_profile_dmnd_ind_transp_el['pf_id'] = df_def_profile_dmnd_ind_transp_el.index.rename('pf_id') + df_def_profile_dmnd_ind_transp_h2.pf_id.max() + 1
df_def_profile_dmnd_ind_transp_el = df_def_profile_dmnd_ind_transp_el[df_def_profile_0.columns]

# Demand el profiles in H2 nodes
# df_def_profile_dmnd_ind_h2_el = df_def_node.nd.copy().rename('primary_nd').reset_index()
# df_def_profile_dmnd_ind_h2_el['pf'] = 'demand_EL_' + df_def_profile_dmnd_ind_h2_el.primary_nd
# df_def_profile_dmnd_ind_h2_el['pf_id'] = df_def_profile_dmnd_ind_h2_el.index.rename('pf_id') + df_def_profile_dmnd_ind_h2.pf_id.max() + 1
# df_def_profile_dmnd_ind_h2_el = df_def_profile_dmnd_ind_h2_el[df_def_profile_0.columns]

# Demand H2 profiles in IND and CH0 node (for the coupling)
df_def_profile_dmnd_ind_ch0_h2_el = pd.concat([df_nd_ind_el,df_nd_ch0_el]).nd.copy().rename('primary_nd').reset_index()
df_def_profile_dmnd_ind_ch0_h2_el['pf'] = 'demand_H2_' + df_def_profile_dmnd_ind_ch0_h2_el.primary_nd
df_def_profile_dmnd_ind_ch0_h2_el['pf_id'] = df_def_profile_dmnd_ind_ch0_h2_el.index.rename('pf_id') + df_def_profile_dmnd_ind_transp_el.pf_id.max() + 1
df_def_profile_dmnd_ind_ch0_h2_el= df_def_profile_dmnd_ind_ch0_h2_el[df_def_profile_0.columns]

# df_def_profile =pd.concat([df_def_profile_dmnd_ind_h2,
#                             df_def_profile_dmnd_ind_h2_el,
#                            ], axis=0)

# df_def_profile =pd.concat([df_def_profile_dmnd_ind_transp_h2,
#                             df_def_profile_dmnd_ind_ch0_h2_el,
#                            ], axis=0)

df_def_profile =pd.concat([df_def_profile_dmnd_ind_transp_h2,
                           df_def_profile_dmnd_ind_transp_el,
                            df_def_profile_dmnd_ind_ch0_h2_el,
                           ], axis=0)

df_def_profile = df_def_profile.reset_index(drop=True)
    
df_def_profile


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NODE_ENCAR for H2 and transport

df_node_encar_0 = pd.read_csv(data_path + '/node_encar.csv')

df_ndca_add_ind_h2 = (dferg_ind_h2.loc[dferg_ind_h2.nd_id_new.isin(df_nd_add.nd), ['nd_id_new', 'erg_tot']]
                         .rename(columns={'erg_tot': 'dmnd_sum', 'nd_id_new': 'nd_id'}))
df_ndca_add_transp_h2 = (dferg_transp_h2.loc[dferg_transp_h2.nd_id_new.isin(df_nd_add.nd), ['nd_id_new', 'erg_tot']]
                         .rename(columns={'erg_tot': 'dmnd_sum', 'nd_id_new': 'nd_id'}))
df_ndca_add_transp_el = (dferg_transp_el.loc[dferg_transp_el.nd_id_new.isin(df_nd_add.nd), ['nd_id_new', 'erg_tot']]
                         .rename(columns={'erg_tot': 'dmnd_sum', 'nd_id_new': 'nd_id'}))

df_ndca_add_ind_el_h2 = pd.DataFrame({'nd_id':df_nd_ind_el.nd.to_list(),
                                      'dmnd_sum':[0,0,0]})
df_ndca_add_ch0_el_h2 = pd.DataFrame({'nd_id':df_nd_ch0_el.nd.to_list(),
                                      'dmnd_sum':0})

data_6 = dict(vc_dmnd_flex=0, ca_id=6, grid_losses=0, grid_losses_absolute=0)
data_0 = dict(vc_dmnd_flex=0, ca_id=0, grid_losses=0, grid_losses_absolute=0)

# Industry H2
df_node_encar_ind_h2_6 = df_ndca_add_ind_h2.assign(**data_6).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
# df_node_encar_ind_h2_6 = pd.merge(df_node_encar_ind_h2_6, df_def_profile_dmnd_ind_h2, left_on='nd_id', right_on='primary_nd', how='inner')
df_node_encar_ind_h2_6 = pd.merge(df_node_encar_ind_h2_6, df_def_profile_dmnd_ind_transp_h2, left_on='nd_id', right_on='primary_nd', how='inner')

# Transport H2
df_node_encar_transp_h2_6 = df_ndca_add_transp_h2.assign(**data_6).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
df_node_encar_transp_h2_6 = pd.merge(df_node_encar_transp_h2_6, df_def_profile_dmnd_ind_transp_h2, left_on='nd_id', right_on='primary_nd', how='inner')

# Transport EL
df_node_encar_transp_el_0 = df_ndca_add_transp_el.assign(**data_0).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
df_node_encar_transp_el_0 = pd.merge(df_node_encar_transp_el_0, df_def_profile_dmnd_ind_transp_el, left_on='nd_id', right_on='primary_nd', how='inner')

# Need to add ca_id = 0 for H2 like for heating
# data_0 = dict(vc_dmnd_flex=0, ca_id=0, grid_losses=0, grid_losses_absolute=0)
# df_node_encar_ind_h2_0 = df_ndca_add_ind_h2.assign(**data_0).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
# df_node_encar_ind_h2_0 = pd.merge(df_node_encar_ind_h2_0, df_def_profile_dmnd_ind_h2_el, left_on='nd_id', right_on='primary_nd', how='inner')
# Industry EL
df_node_encar_ind_el_6 = df_ndca_add_ind_el_h2.assign(**data_6).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
# df_node_encar_ind_el_6 = pd.merge(df_node_encar_ind_el_6, df_def_profile_dmnd_ind_h2_el, left_on='nd_id', right_on='primary_nd', how='inner')
df_node_encar_ind_el_6 = pd.merge(df_node_encar_ind_el_6, df_def_profile_dmnd_ind_ch0_h2_el, left_on='nd_id', right_on='primary_nd', how='inner')

# Transport CH0
df_node_encar_ch0_el_6 = df_ndca_add_ch0_el_h2.assign(**data_6).reindex(columns=df_node_encar_0.columns).reset_index(drop=True)
df_node_encar_ch0_el_6 = pd.merge(df_node_encar_ch0_el_6, df_def_profile_dmnd_ind_ch0_h2_el, left_on='nd_id', right_on='primary_nd', how='inner')



df_node_encar_ind_h2 = pd.concat([df_node_encar_ind_h2_6,
                                  # df_node_encar_ind_h2_0,
                                  df_node_encar_ind_el_6,
                                  ]).reset_index(drop=True)

list_dmnd = [c for c in df_node_encar_ind_h2 if 'dmnd_sum' in c]

df_node_encar_ind_h2 = df_node_encar_ind_h2.assign(**{c: df_node_encar_ind_h2.dmnd_sum
                                        for c in list_dmnd})

#  Here be careful if the input data are for 2050 or for 2015

fct_dmnd = pd.read_csv(base_dir+'/h2_transport/demand/factor_dmnd_transp_future_years.csv',sep=';')

fct_dmnd = pd.read_csv(base_dir+'/h2_transport/demand/factor_dmnd_transp_future_years_2050.csv',sep=';')


df_node_encar_transp_h2 = pd.concat([df_node_encar_transp_h2_6,
                                  # df_node_encar_ind_h2_0,
                                  df_node_encar_ch0_el_6,
                                  ]).reset_index(drop=True)

list_dmnd = [c for c in df_node_encar_transp_h2 if 'dmnd_sum' in c]

df_node_encar_transp_h2 = df_node_encar_transp_h2.assign(**{c: df_node_encar_transp_h2.dmnd_sum
                                        for c in list_dmnd})

#  For now we use 2050, so no need to factor it

# df = df_node_encar_transp_h2.filter(like='dmnd_sum')*fct_dmnd
# df_node_encar_transp_h2.update(df)

df_node_encar_transp_el = df_node_encar_transp_el_0

list_dmnd = [c for c in df_node_encar_transp_el if 'dmnd_sum' in c]

df_node_encar_transp_el = df_node_encar_transp_el.assign(**{c: df_node_encar_transp_el.dmnd_sum
                                        for c in list_dmnd})

# df_node_encar_ind_h2['dmnd_pf_id'] = df_node_encar_ind_h2.pf
# df_node_encar_ind_h2 = df_node_encar_ind_h2.loc[:, df_node_encar_0.columns]

# # for df, idx in [(df_def_node, 'nd'), (df_def_profile, ['pf', 'dmnd_pf'])]:
# #     df_node_encar_ind_h2, _ = translate_id(df_node_encar_ind_h2, df, idx)

# for df, idx in [(pd.concat([df_def_node_0,df_def_node]), 'nd'), (df_def_profile, ['pf', 'dmnd_pf'])]:
#     df_node_encar_ind_h2, _ = translate_id(df_node_encar_ind_h2, df, idx)

# df_node_encar_h2 = pd.concat([df_node_encar_ind_h2,df_node_encar_transp_h2]).reset_index(drop=True)
    
# df_node_encar_h2['dmnd_pf_id'] = df_node_encar_h2.pf
# df_node_encar_h2 = df_node_encar_h2.loc[:, df_node_encar_0.columns]

df_node_encar_h2_el = pd.concat([df_node_encar_ind_h2,df_node_encar_transp_h2,df_node_encar_transp_el]).reset_index(drop=True)
    
df_node_encar_h2_el['dmnd_pf_id'] = df_node_encar_h2_el.pf
df_node_encar_h2_el = df_node_encar_h2_el.loc[:, df_node_encar_0.columns]

# for df, idx in [(df_def_node, 'nd'), (df_def_profile, ['pf', 'dmnd_pf'])]:
#     df_node_encar_ind_h2, _ = translate_id(df_node_encar_ind_h2, df, idx)

# for df, idx in [(pd.concat([df_def_node_0,df_def_node]), 'nd'), (df_def_profile, ['pf', 'dmnd_pf'])]:
#     df_node_encar_ind_h2, _ = translate_id(df_node_encar_ind_h2, df, idx)

# for df, idx in [(pd.concat([df_def_node_0,df_def_node]), 'nd'), (df_def_profile, ['pf', 'dmnd_pf'])]:
#     df_node_encar_h2, _ = translate_id(df_node_encar_h2, df, idx)

for df, idx in [(pd.concat([df_def_node_0,df_def_node]), 'nd'), (df_def_profile, ['pf', 'dmnd_pf'])]:
    df_node_encar_h2_el, _ = translate_id(df_node_encar_h2_el, df, idx)


# df_node_encar_add = pd.concat([df_node_encar_ind_h2,
#                                ])

# df_node_encar_add = pd.concat([df_node_encar_h2,
#                                ])
df_node_encar_add = pd.concat([df_node_encar_h2_el,
                               ])
#df_node_encar_dhw.update(df_node_encar_wo_dhw)


# df_node_encar_new = df_node_encar_add.reset_index(drop=True)
df_node_encar = df_node_encar_add.reset_index(drop=True)




# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFDMND for H2 and transport

df_tm_st =pd.read_csv(base_dir+'/h2_transport/timemap/timestamp_template.csv')
df_tm_st['datetime'] = df_tm_st['datetime'].astype('datetime64[ns]')

df_profdmnd_0 = pd.read_csv(data_path + '/profdmnd.csv').head()

# H2 industry                                              
df_dmnd_ind_h2_add = dfload_ind_h2.copy()

df_dmnd_ind_h2_add['ca_id'] = 6

# df_dmnd_ind_h2_add = pd.merge(df_dmnd_ind_h2_add, df_def_profile_dmnd_ind_h2[['pf_id', 'primary_nd']], left_on='nd_id', right_on='primary_nd')
df_dmnd_ind_h2_add = pd.merge(df_dmnd_ind_h2_add, df_def_profile_dmnd_ind_transp_h2[['pf_id', 'primary_nd']], left_on='nd_id', right_on='primary_nd')
df_dmnd_ind_h2_add = df_dmnd_ind_h2_add.rename(columns={'erg_tot': 'value', 'pf_id': 'dmnd_pf_id'})
df_dmnd_ind_h2_add['nd_id'] = df_dmnd_ind_h2_add.nd_id.replace(dict_nd_id_ind_h2)
df_dmnd_ind_h2_add['hy'] = 24*df_dmnd_ind_h2_add.doy - 24

df_dmnd_ind_h2_add['erg_tot_fossil'] = 0
df_dmnd_ind_h2_add['erg_tot_retr_1pc'] = 0
df_dmnd_ind_h2_add['erg_tot_retr_2pc'] = 0
df_dmnd_ind_h2_add['erg_tot_retr_1pc_fossil'] = 0
df_dmnd_ind_h2_add['erg_tot_retr_2pc_fossil'] = 0

# H2 transport                                            
df_dmnd_transp_h2_add = dfload_transp_h2.copy()

df_dmnd_transp_h2_add['ca_id'] = 6

df_dmnd_transp_h2_add = pd.merge(df_dmnd_transp_h2_add, df_def_profile_dmnd_ind_transp_h2[['pf_id', 'primary_nd']], left_on='nd_id', right_on='primary_nd')
df_dmnd_transp_h2_add = df_dmnd_transp_h2_add.rename(columns={'erg_tot': 'value', 'pf_id': 'dmnd_pf_id'})
df_dmnd_transp_h2_add['nd_id'] = df_dmnd_transp_h2_add.nd_id.replace(dict_nd_id_transp_h2)
df_dmnd_transp_h2_add['hy'] = 24*df_dmnd_transp_h2_add.doy - 24

df_dmnd_transp_h2_add['erg_tot_fossil'] = 0
df_dmnd_transp_h2_add['erg_tot_retr_1pc'] = 0
df_dmnd_transp_h2_add['erg_tot_retr_2pc'] = 0
df_dmnd_transp_h2_add['erg_tot_retr_1pc_fossil'] = 0
df_dmnd_transp_h2_add['erg_tot_retr_2pc_fossil'] = 0


# EL transport                                            
df_dmnd_transp_el_add = dfload_transp_el.copy()

df_dmnd_transp_el_add['ca_id'] = 6

df_dmnd_transp_el_add = pd.merge(df_dmnd_transp_el_add, df_def_profile_dmnd_ind_transp_el[['pf_id', 'primary_nd']], left_on='nd_id', right_on='primary_nd')
df_dmnd_transp_el_add = df_dmnd_transp_el_add.rename(columns={'erg_tot': 'value', 'pf_id': 'dmnd_pf_id'})
df_dmnd_transp_el_add['nd_id'] = df_dmnd_transp_el_add.nd_id.replace(dict_nd_id_transp_el)
df_dmnd_transp_el_add['hy'] = 24*df_dmnd_transp_el_add.doy - 24

df_dmnd_transp_el_add['erg_tot_fossil'] = 0
df_dmnd_transp_el_add['erg_tot_retr_1pc'] = 0
df_dmnd_transp_el_add['erg_tot_retr_2pc'] = 0
df_dmnd_transp_el_add['erg_tot_retr_1pc_fossil'] = 0
df_dmnd_transp_el_add['erg_tot_retr_2pc_fossil'] = 0


df_dmnd_add = pd.concat([df_dmnd_ind_h2_add,
                         df_dmnd_transp_h2_add,
                         df_dmnd_transp_el_add,
                         ])


df_profdmnd_add = df_dmnd_add[df_profdmnd_0.columns.tolist()].reset_index(drop=True)


#df_profdmnd_new = pd.concat([df_profdmnd_0,df_profdmnd_dhw])#,df_profdmnd_ht_el])
# Without DHW only
# df_profdmnd_new = pd.concat([df_profdmnd_0_wo_dhw,df_profdmnd_add])
# Without DHW and DSR
# df_profdmnd_new = pd.concat([df_profdmnd_0_wo_dhw_dsr,df_profdmnd_add])

# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFPRICE

# --> NO CHANGES! HOUSEHOLDS USE CH0 PRICE PROFILES


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PROFSUPPLY

# --> NO CHANGES!


# %% ~~~~~~~~~~~~~~~~~~~~~~~ PLANT_ENCAR (needs profsupply data)

dict_pp_new = pd.Series(df_def_plant.pp_id.values,index=df_def_plant.pp).to_dict()
# dict_nd_id_all = dict(pd.Series(df_def_node_0.nd_id.values,index=df_def_node_0.nd).to_dict(), **dict_nd_id)
dict_pt_id_all = dict(pd.Series(df_def_pp_type_0.pt_id.values,index=df_def_pp_type_0.pt).to_dict(),
                      **pd.Series(df_def_pp_type.pt_id.values,index=df_def_pp_type.pt))

df_plant_encar = pd.read_csv(data_path + '/plant_encar.csv')

# dfplant_costs = pd.read_csv(base_dir + '/h2_transport/costs/cost_tech_h2_ely_sto_fc.csv',sep=';')
# dfplant_costs = pd.read_csv(base_dir + '/h2_transport/costs/cost_tech_h2_ely_sto_fc_ind_el_nodes.csv',sep=';')
# dfplant_costs = pd.read_csv(base_dir + '/h2_transport/costs/cost_tech_h2_ely_sto_fc_ind_ch0_el_nodes.csv',sep=';')
# dfplant_costs = pd.read_csv(base_dir + '/h2_transport/costs/cost_tech_h2_ely_sto_fc_ind_ch0_el_nodes_future.csv',sep=';')
dfplant_costs = pd.read_csv(base_dir + '/h2_transport/costs/cost_tech_h2_ely_sto_fc_ind_ch0_el_nodes_future_size_dependent.csv',sep=';')

df_ppca_add = dfplant_costs.copy()

# df_ppca_add['nd_id'] = df_ppca_add.nd_id.replace(dict_nd_id_ind_h2)
# df_ppca_add['nd_id'] = df_ppca_add.nd_id.replace(dict_nd_id_ind_el)
df_ppca_add['nd_id'] = df_ppca_add.nd_id.replace(dict_nd_id_ind_ch0_el)


df_ppca_add['pt_id'] = df_ppca_add.pt_id.replace(dict_pt_id_all)

df_ppca_add = pd.merge(df_ppca_add, df_def_plant[['pp_id','pt_id','nd_id']], on= ['pt_id','nd_id'])
df_ppca_add = df_ppca_add.drop(columns=['nd_id','pt_id'])

df_ppca_add = df_ppca_add.assign(
                                 factor_lin_0=0, factor_lin_1=0,
                                 cap_avlb=1, vc_ramp=0,
                                 vc_om=0, erg_chp=None)


df_plant_encar_1 = pd.DataFrame()
df_plant_encar_1 = pd.concat([df_plant_encar, df_ppca_add])

list_cap = [c for c in df_plant_encar.columns if 'cap_pwr_leg' in c]

df_plant_encar_2 = df_plant_encar_1.loc[df_plant_encar_1.pp_id.isin(df_pp_add.pp_id)].assign(**{cap: 0 for cap in list_cap}).set_index('pp_id')

df_plant_encar_1 = df_plant_encar_1.set_index('pp_id')
df_plant_encar_1.update(df_plant_encar_2) 
df_plant_encar_1 = df_plant_encar_1.reset_index()





df_plant_encar_new = df_plant_encar_1.reset_index(drop=True)#(drop=True)  

# %% ~~~~~~~~~~~~~~~~~~~~ NODE_CONNECT

# df_node_connect_0 = pd.read_csv(data_path + '/node_connect.csv').query(
#                         'nd_id in %s and nd_2_id not in %s'%(
#                         df_nd_ch0_el.nd_id.tolist(),df_nd_not_res.nd_id.tolist())).reset_index(drop=True)
df_node_connect_0 = pd.read_csv(data_path + '/node_connect.csv').head()

# df_node_connect = aql.read_sql(db, sc, 'node_connect',
#                                filt=[('nd_id', df_nd_ch0_el.nd_id.tolist(),  ' = ', ' AND '),
#                                      ('nd_2_id', df_nd_not_res.nd_id.tolist(), ' != ', ' AND ')])


node_ind_rur = df_def_node_0.loc[df_def_node_0.nd.str.contains('IND_RUR')].nd.values
node_ind_sub = df_def_node_0.loc[df_def_node_0.nd.str.contains('IND_SUB')].nd.values
node_ind_urb = df_def_node_0.loc[df_def_node_0.nd.str.contains('IND_URB')].nd.values

node_oco_rur = df_def_node_0.loc[df_def_node_0.nd.str.contains('OCO_RUR')].nd.values
node_oco_sub = df_def_node_0.loc[df_def_node_0.nd.str.contains('OCO_SUB')].nd.values
node_oco_urb = df_def_node_0.loc[df_def_node_0.nd.str.contains('OCO_URB')].nd.values

node_ind_h2 = df_def_node.loc[df_def_node.nd.str.contains('IND')].nd.values

node_ch0 = df_def_node_0.loc[df_def_node_0.nd.str.contains('CH0')].nd.values
node_transp_h2 = df_def_node.loc[df_def_node.nd.str.contains('TRANSP_H2')].nd.values
node_transp_el = df_def_node.loc[df_def_node.nd.str.contains('TRANSP_EL')].nd.values

#  Connection H2 for Ind
data_ind_rur_h2 = dict(nd_id=node_ind_rur, nd_2_id=node_ind_h2, ca_id=6, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_ind_sub_h2 = dict(nd_id=node_ind_sub, nd_2_id=node_ind_h2, ca_id=6, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_ind_urb_h2 = dict(nd_id=node_ind_urb, nd_2_id=node_ind_h2, ca_id=6, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)

# Connection H2 for Transport

data_ch0 = dict(nd_id=node_ch0, nd_2_id=node_transp_h2, ca_id=6, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)

# Connection El for Transport
data_oco_rur = dict(nd_id=node_oco_rur, nd_2_id=node_transp_el, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_oco_sub = dict(nd_id=node_oco_sub, nd_2_id=node_transp_el, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_oco_urb = dict(nd_id=node_oco_urb, nd_2_id=node_transp_el, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)

data_ind_rur = dict(nd_id=node_ind_rur, nd_2_id=node_transp_el, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_ind_sub = dict(nd_id=node_ind_sub, nd_2_id=node_transp_el, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)
data_ind_urb = dict(nd_id=node_ind_urb, nd_2_id=node_transp_el, ca_id=0, mt_id='all',cap_trme_leg=9e9,cap_trmi_leg=0.0)


# data_ind_df = pd.concat([pd.DataFrame(data_ind_rur),pd.DataFrame(data_ind_sub),pd.DataFrame(data_ind_urb)])
# data_ind_df = expand_rows(data_ind_df, ['mt_id'], [range(12)])
# data_ind_df[['mt_id']] = data_ind_df.mt_id.astype(int)
# df_node_connect = data_ind_df[df_node_connect_0.columns]

data_ind_ch0_df = pd.concat([pd.DataFrame(data_ind_rur_h2),pd.DataFrame(data_ind_sub_h2),pd.DataFrame(data_ind_urb_h2),
                             pd.DataFrame(data_ch0),
                             pd.DataFrame(data_ind_rur),pd.DataFrame(data_ind_sub),pd.DataFrame(data_ind_urb),
                             pd.DataFrame(data_oco_rur),pd.DataFrame(data_oco_sub),pd.DataFrame(data_oco_urb)])

data_ind_ch0_df = expand_rows(data_ind_ch0_df, ['mt_id'], [range(12)])
data_ind_ch0_df[['mt_id']] = data_ind_ch0_df.mt_id.astype(int)
df_node_connect = data_ind_ch0_df[df_node_connect_0.columns]


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






