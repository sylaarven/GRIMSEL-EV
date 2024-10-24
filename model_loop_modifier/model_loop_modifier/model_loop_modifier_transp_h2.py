#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:54:18 2019

@author: arthurrinaldi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:23:06 2018

@author: martin-c-s
"""
import sys, os, socket
import pandas as pd
import numpy as np
import logging
from glob import glob

import pyomo.environ as po

from grimsel.core.io import IO
import grimsel.auxiliary.sqlutils.aux_sql_func as aql
from grimsel.auxiliary.aux_m_func import cols2tuplelist
import grimsel.auxiliary.maps as maps
from grimsel import _get_logger

import grimsel_config as config

import multiprocess as mp

logger = _get_logger(__name__)
logger.setLevel('DEBUG')


class ModelLoopModifier():
    '''
    The purpose of this class is to modify the parameters of the BaseModel
    objects in dependence on the current run_id.

    This might include the modification of parameters like CO2 prices,
    capacities, but also more profound modifications like the
    redefinition of constraints.
    '''

    def __init__(self, ml):
        '''
        To initialize, the ModelBase object is made an instance attribute.
        '''

        self.ml = ml

    def select_swiss_scenarios(self, dict_ch=None):

        if dict_ch is None:
            dict_ch = {0: 'default', 1: 'hi_gas', 2: 'hi_ren'}

        slct_ch = dict_ch[self.ml.dct_step['swch']]

        self.ml.dct_vl['swch_vl'] = slct_ch

        return slct_ch

    def select_hp_scenarios(self, dict_hp=None): 
        
        ''' Select heat-pumps scenario for archetypes for future years '''
        
        if dict_hp is None:
            dict_hp = {0: 'bau', 1: 'fossil', 2: 'full'}

        slct_hp = dict_hp[self.ml.dct_step['swhp']]

        self.ml.dct_vl['swhp_vl'] = slct_hp

        return slct_hp
    
    def select_transmission_scenarios(self, dict_tr=None): 
        
        ''' Select transmission scenario for archetypes for future years '''
        
        if dict_tr is None:
            dict_tr = {0: 'default', 1: 'min_5pc', 2: 'plus_5pc', 3: 'plus_10pc', 4: 'plus_15pc'}

        slct_tr = dict_tr[self.ml.dct_step['swtr']]

        self.ml.dct_vl['swtr_vl'] = slct_tr

        return slct_tr
    
    def select_retrofit_scenarios(self, dict_rf=None): 
        
        ''' Select retrofit scenario for archetypes for future years '''
        
        if dict_rf is None:
            dict_rf = {0: 'default', 1: 'retr_1pc', 2: 'retr_2pc'}

        slct_rf = dict_rf[self.ml.dct_step['swrf']]

        self.ml.dct_vl['swrf_vl'] = slct_rf

        return slct_rf
    
    def select_demand_profile_res(self, dict_dpf=None): 
        
        ''' Select demand profile (EE), for EL archetypes for future years '''
        
        if dict_dpf is None:
            dict_dpf = {0: 'default', 1: 'ee', 2: 'best'}

        slct_dpf = dict_dpf[self.ml.dct_step['swdpf']]

        self.ml.dct_vl['swdpf_vl'] = slct_dpf

        return slct_dpf

    # def select_ev_scenarios(self, dict_ev=None): 
        
    #     ''' Select EV scenario for archetypes for future years '''
        
    #     if dict_ev is None:
    #         dict_ev = {0: 'basis', 1: 'ss', 2:'full', 3: 'bass'}

    #     slct_ev = dict_ev[self.ml.dct_step['swev']]

    #     self.ml.dct_vl['swev_vl'] = slct_ev

    #     return slct_ev
    
    def select_h2_transp_scenarios(self, dict_h2tr=None): 
        
        # ''' Select scenario for H2 and EL penetration to freight transport'''
        ''' Select scenario for H2 penetration to in each transport categories'''
        
        if dict_h2tr is None:

           dict_h2tr = {0:	'2050+',
                        1:	'100H2_all',
                        2:	'100EL_all',
                        3:	'100HCV_0LCV_0PC',
                        4:	'100HCV_25LCV_0PC',
                        5:	'100HCV_50LCV_0PC',
                        6:	'100HCV_75LCV_0PC',
                        7:	'100HCV_100LCV_0PC',
                        8:	'100HCV_0LCV_25PC',
                        9:	'100HCV_25LCV_25PC',
                        10:	'100HCV_50LCV_25PC',
                        11:	'100HCV_75LCV_25PC',
                        12:	'100HCV_100LCV_25PC',
                        13:	'100HCV_0LCV_50PC',
                        14:	'100HCV_25LCV_50PC',
                        15:	'100HCV_50LCV_50PC',
                        16:	'100HCV_75LCV_50PC',
                        17:	'100HCV_100LCV_50PC',
                        18:	'100HCV_0LCV_75PC',
                        19:	'100HCV_25LCV_75PC',
                        20:	'100HCV_50LCV_75PC',
                        21:	'100HCV_75LCV_75PC',
                        22:	'100HCV_100LCV_75PC',
                        23:	'100HCV_0LCV_100PC',
                        24:	'100HCV_25LCV_100PC',
                        25:	'100HCV_50LCV_100PC',
                        26:	'100HCV_75LCV_100PC',
                        27:	'75HCV_0LCV_0PC',
                        28:	'75HCV_25LCV_0PC',
                        29:	'75HCV_50LCV_0PC',
                        30:	'75HCV_75LCV_0PC',
                        31:	'75HCV_100LCV_0PC',
                        32:	'75HCV_0LCV_25PC',
                        33:	'75HCV_25LCV_25PC',
                        34:	'75HCV_50LCV_25PC',
                        35:	'75HCV_75LCV_25PC',
                        36:	'75HCV_100LCV_25PC',
                        37:	'75HCV_0LCV_50PC',
                        38:	'75HCV_25LCV_50PC',
                        39:	'75HCV_50LCV_50PC',
                        40:	'75HCV_75LCV_50PC',
                        41:	'75HCV_100LCV_50PC',
                        42:	'75HCV_0LCV_75PC',
                        43:	'75HCV_25LCV_75PC',
                        44:	'75HCV_50LCV_75PC',
                        45:	'75HCV_75LCV_75PC',
                        46:	'75HCV_100LCV_75PC',
                        47:	'75HCV_0LCV_100PC',
                        48:	'75HCV_25LCV_100PC',
                        49:	'75HCV_50LCV_100PC',
                        50:	'75HCV_75LCV_100PC',
                        51: '75HCV_100LCV_100PC',
                        52:	'50HCV_0LCV_0PC',
                        53:	'50HCV_25LCV_0PC',
                        54:	'50HCV_50LCV_0PC',
                        55:	'50HCV_75LCV_0PC',
                        56:	'50HCV_100LCV_0PC',
                        57:	'50HCV_0LCV_25PC',
                        58:	'50HCV_25LCV_25PC',
                        59:	'50HCV_50LCV_25PC',
                        60:	'50HCV_75LCV_25PC',
                        61:	'50HCV_100LCV_25PC',
                        62:	'50HCV_0LCV_50PC',
                        63:	'50HCV_25LCV_50PC',
                        64:	'50HCV_50LCV_50PC',
                        65:	'50HCV_75LCV_50PC',
                        66:	'50HCV_100LCV_50PC',
                        67:	'50HCV_0LCV_75PC',
                        68:	'50HCV_25LCV_75PC',
                        69:	'50HCV_50LCV_75PC',
                        70:	'50HCV_75LCV_75PC',
                        71:	'50HCV_100LCV_75PC',
                        72:	'50HCV_0LCV_100PC',
                        73:	'50HCV_25LCV_100PC',
                        74:	'50HCV_50LCV_100PC',
                        75:	'50HCV_75LCV_100PC',
                        76:	'50HCV_100LCV_100PC',
                        77:	'25HCV_0LCV_0PC',
                        78:	'25HCV_25LCV_0PC',
                        79:	'25HCV_50LCV_0PC',
                        80:	'25HCV_75LCV_0PC',
                        81:	'25HCV_100LCV_0PC',
                        82:	'25HCV_0LCV_25PC',
                        83:	'25HCV_25LCV_25PC',
                        84:	'25HCV_50LCV_25PC',
                        85:	'25HCV_75LCV_25PC',
                        86:	'25HCV_100LCV_25PC',
                        87:	'25HCV_0LCV_50PC',
                        88:	'25HCV_25LCV_50PC',
                        89:	'25HCV_50LCV_50PC',
                        90:	'25HCV_75LCV_50PC',
                        91:	'25HCV_100LCV_50PC',
                        92:	'25HCV_0LCV_75PC',
                        93:	'25HCV_25LCV_75PC',
                        94:	'25HCV_50LCV_75PC',
                        95:	'25HCV_75LCV_75PC',
                        96:	'25HCV_100LCV_75PC',
                        97:	'25HCV_0LCV_100PC',
                        98:	'25HCV_25LCV_100PC',
                        99:	'25HCV_50LCV_100PC',
                        100:'25HCV_75LCV_100PC',
                        101:'25HCV_100LCV_100PC',
                        102:'0HCV_25LCV_0PC',
                        103:'0HCV_50LCV_0PC',
                        104:'0HCV_75LCV_0PC',
                        105	:'0HCV_100LCV_0PC',
                        106:'0HCV_0LCV_25PC',
                        107:'0HCV_25LCV_25PC',
                        108:'0HCV_50LCV_25PC',
                        109:'0HCV_75LCV_25PC',
                        110:'0HCV_100LCV_25PC',
                        111:'0HCV_0LCV_50PC',
                        112:'0HCV_25LCV_50PC',
                        113:'0HCV_50LCV_50PC',
                        114:'0HCV_75LCV_50PC',
                        115:'0HCV_100LCV_50PC',
                        116:'0HCV_0LCV_75PC',
                        117:'0HCV_25LCV_75PC',
                        118:'0HCV_50LCV_75PC',
                        119:'0HCV_75LCV_75PC',
                        120:'0HCV_100LCV_75PC',
                        121:'0HCV_0LCV_100PC',
                        122:'0HCV_25LCV_100PC',
                        123:'0HCV_50LCV_100PC',
                        124:'0HCV_75LCV_100PC',
                        125:'0HCV_100LCV_100PC',
                        }
            # dict_h2tr = {0: '2050+', 
            #              1: 'full_h2', 2:'full_el',
            #              3: 'full_h2_LCV_HCV', 4: 'full_el_PC_LCV',
            #              5: '50%h2LCV_'}

        slct_h2tr = dict_h2tr[self.ml.dct_step['swh2tr']]

        self.ml.dct_vl['swh2tr_vl'] = slct_h2tr

        return slct_h2tr
    
    def set_future_year(self, slct_ch, slct_tr, slct_hp, slct_rf, slct_dpf, slct_h2tr, dict_fy=None):
    # def set_future_year(self, slct_ch, slct_tr, slct_hp, slct_rf, slct_dpf, slct_ev, slct_h2tr, dict_fy=None):
        
        list_hp_aw = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('HP_AW')].pp_id.tolist()
        list_hp_ww = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('HP_WW')].pp_id.tolist()
        list_dhw_aw = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_AW')].pp_id.tolist()
        list_dhw_ww = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_WW')].pp_id.tolist()
        
        list_dhw_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_STO')].pp_id.tolist()
        list_dhw_bo = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_BO')].pp_id.tolist()

        list_all_add = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.set_def_add == 1].pp_id.tolist()
        
        # nd_arch = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH|OCO|IND')].nd_id.tolist()
        nd_arch = self.ml.m.df_def_node.loc[np.invert(self.ml.m.df_def_node.nd.isin(['CH0', 'AT0', 'IT0', 'FR0', 'DE0','IND_H2','TRANSP_H2','TRANSP_EL']))].nd_id.tolist()
        nd_nat = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('CH0|AT0|IT0|FR0|DE0')].nd_id.tolist()
        nd_arch_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('HT')].nd_id.tolist()
        nd_arch_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH')].nd_id.tolist()
        nd_arch_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('DSR')].nd_id.tolist()
        nd_arch_ev = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('EV')].nd_id.tolist()
        nd_arch_el = list(set(nd_arch) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev))
        nd_arch_el_res = list(set(nd_arch_res) - set(nd_arch_ht) - set(nd_arch_dsr) - set(nd_arch_ev))
        nd_ind_h2 = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('IND_H2')].nd_id.tolist()
        nd_arch_non_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('OCO|IND')].nd_id.tolist()
        nd_arch_el_non_res = list(set(nd_arch_non_res) - set(nd_ind_h2))
        nd_transp_h2 = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('TRANSP_H2')].nd_id.tolist()
        nd_transp_el = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('TRANSP_EL')].nd_id.tolist()

        if nd_arch_el:    
            tm_id_el = self.ml.m.dict_nd_tm_id[nd_arch_el[0]]
        else:
            pass
            # tm_id_el = 0
        if nd_arch_ht:    
            tm_id_ht = self.ml.m.dict_nd_tm_id[nd_arch_ht[0]]
        else:
            # pass
            tm_id_ht = 1 # Here it's juast to avoid some error on some test when not selecting heating nodes
        if nd_arch_dsr:    
            tm_id_dsr = self.ml.m.dict_nd_tm_id[nd_arch_dsr[0]]
        else:
            pass
            tm_id_dsr = 1 # Here it's juast to avoid some error on some test when not selecting dsr nodes
        if nd_ind_h2:    
            tm_id_ind_h2 = self.ml.m.dict_nd_tm_id[nd_ind_h2[0]]
        else:
            pass
        if nd_transp_h2:    
            tm_id_transp_h2 = self.ml.m.dict_nd_tm_id[nd_transp_h2[0]]
        else:
            pass
        if nd_transp_el:    
            tm_id_transp_el = self.ml.m.dict_nd_tm_id[nd_transp_el[0]]
        else:
            pass

        if dict_fy is None:
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }


        rng_fy = np.arange(len(dict_fy))
        rng_fyp = np.ones(len(rng_fy)) * {val: key for key, val in dict_fy.items()}[2015]
        dict_previous = dict(np.array([rng_fy, rng_fyp]).T)


        slct_fy = self.ml.dct_step['swfy']

        str_fy = '_yr' + str(dict_fy[slct_fy]) if dict_fy[slct_fy] != 2015 else ''
        str_fyp = '_yr' + str(dict_fy[dict_previous[slct_fy]]) if dict_fy[dict_previous[slct_fy]] != 2015 else ''

        #######################################################################
        def set_fuel_prices(str_fy=None):
            ''' Select fuel price values for selected year. '''

            if not str_fy:
                str_fy = ''

            slct_col = 'vc_fl' + str_fy
            msg = ('Setting vc_fl to values' + str_fy.replace('_', ' ') +
                   ' from column {}'.format(slct_col))
            logger.info(msg)

            par = self.ml.m.dict_par['vc_fl']
            df_new = self.ml.m.df_fuel_node_encar[['fl_id', 'nd_id', slct_col]]
            df_new = df_new.rename(columns={slct_col: 'vc_fl'})
            col_mt_fact = 'mt_fact' if not str_fy else 'mt_fact_others'
            par.init_update(df_new, col_mt_fact)

        # def set_curtailment_prices(str_fy=None):
        #     ''' Select curtailement price values for selected year. '''

        #     if not str_fy:
        #         str_fy = ''

        #     slct_col = 'vc_curt' + str_fy
        #     msg = ('Setting vc_curt to values' + str_fy.replace('_', ' ') +
        #            ' from column {}'.format(slct_col))
        #     logger.info(msg)

        #     par = self.ml.m.dict_par['vc_curt']
        #     df_new = self.ml.m.df_fuel_node_encar[['fl_id', 'nd_id', slct_col]]
        #     df_new = df_new.rename(columns={slct_col: 'vc_curt'})
        #     col_mt_fact = 'mt_fact' if not str_fy else 'mt_fact_others'
        #     par.init_update(df_new, col_mt_fact)


        #######################################################################
        def set_cap_pwr_leg(slct_ch, str_fy=None, excl_pp=[]):
            ''' Select power plant capacities for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'cap_pwr_leg' + str_fy
            msg = ('Setting cap_pwr_leg to values' + str_fy.replace('_', ' ') +
                   ' from column {}'.format(slct_col))
            logger.info(msg)

            mask_excl = -self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
            dct_cap = (self.ml.m.df_plant_encar.loc[mask_excl]
                             .set_index(['pp_id', 'ca_id'])[slct_col]
                             .to_dict())
        
            if slct_ch != 'default':

                df = self.ml.m.df_plant_encar_scenarios.copy()
                df = df.loc[df.scenario == slct_ch]

                dct_cap_scen = (df.set_index(['pp_id', 'ca_id'])[slct_col]
                                  .to_dict())

                dct_cap.update(dct_cap_scen)
                

                
            for kk, vv in dct_cap.items():
                self.ml.m.cap_pwr_leg[kk] = vv


        #######################################################################
        def set_cap_avlb(str_fy=None):

            if str_fy == None:
                str_fy = ''

            col_mt_fact = 'mt_fact' if not str_fy else 'mt_fact_others'

            msg = ('Setting cap_avlb monthly adjustment to values'
                   + ' from column {}'.format(col_mt_fact))

            logger.info(msg)

            par = self.ml.m.dict_par['cap_avlb']
            mask_pp = self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.pp)
            df_new = self.ml.m.df_plant_encar.loc[mask_pp,
                                                  ['pp_id', 'ca_id', 'cap_avlb']]

            par.init_update(df_new, col_mt_fact)

            
        #######################################################################
        def set_dmnd(slct_hp, slct_rf, slct_dpf, slct_h2tr, str_fy=None): # slct_ev,
            ''' Scale demand profiles for selected year. '''

            if str_fy == None:
                str_fy = ''
            
            slct_col = 'dmnd_sum' + str_fy
            slct_col_prev = 'dmnd_sum' + str_fyp
            # msg = ('Setting demand normal dmnd_sum to values' + str_fy.replace('_', ' ')
            #       + ' from column {}'.format(slct_col))

            # logger.info(msg)
            last_dmnd = self.ml.m.df_node_encar.set_index(['nd_id', 'ca_id'])[slct_col_prev]
            next_dmnd = self.ml.m.df_node_encar.set_index(['nd_id', 'ca_id'])[slct_col]
            
            dmnd_2050 = self.ml.m.df_node_encar.set_index(['nd_id', 'ca_id'])['dmnd_sum_yr2050']
            dmnd_2015 = self.ml.m.df_node_encar.set_index(['nd_id', 'ca_id'])['dmnd_sum']
            scaling_factor = (next_dmnd / last_dmnd).rename('scale')
            scaling_factor_ht = (next_dmnd / dmnd_2050).rename('scale')*0.882151# Climate correction
            scaling_factor_dhw = (next_dmnd / dmnd_2050).rename('scale')# No Climate correction
            
            # reverse scaling factor for transport -> NO need as 2050 input data
            # scaling_factor_h2tr = (dmnd_2050 / dmnd_2015).rename('scale')

            # Non RES scaling 
            # df_el = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & ~(
            #                                                 IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res+nd_arch_dsr))]
            df_el = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (
                                                            IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_nat+nd_arch_el_non_res))]
            df_el = df_el.join(scaling_factor, on=scaling_factor.index.names)
            df_el['scale'] = df_el['scale'].fillna(0) 
            df_el['value_new'] = df_el.value * df_el.scale
            
            # Inlcude DSR node as not changing (default ee scen)
            df_el_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (
                                                            IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res+nd_arch_dsr))]
            
            # RES with no scaling factor and non RES with scaling factor
            dct_dmnd_el = (df_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
            dct_dmnd_el_res = (df_el_res.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
            
            for kk, vv in dct_dmnd_el.items():
                self.ml.m.dmnd[kk] = vv
            
            for kk, vv in dct_dmnd_el_res.items():
                self.ml.m.dmnd[kk] = vv
                            
            # H2 demand starting in 2020 or 2030
            msg = ('Setting H2 demand INDUSTRY to 0 for year ' + str_fy.replace('_', ' '))
            msg_tr = ('Setting H2 demand TRANSP to 0 for year ' + str_fy.replace('_', ' '))
            msg_tr_el = ('Setting EL demand TRANSP to 0 for year ' + str_fy.replace('_', ' '))
            msg_tr_ev = ('Setting EV demand residential to 0 for year ' + str_fy.replace('_', ' '))

            msg_ind = ('Setting H2 demand INDUSTRY to normal values for year ' + str_fy.replace('_', ' '))
            msg_transp = ('Setting H2 demand TRANSP to values for year ' + str_fy.replace('_', ' ')
                          + ' for transport scenario ' + slct_h2tr)
            msg_transp_el = ('Setting EL demand TRANSP to values for year ' + str_fy.replace('_', ' ')
                          + ' for transport scenario ' + slct_h2tr)
            msg_ev = ('Setting EV demand residential for year ' + str_fy.replace('_', ' ')
                      + ' for transport scenario ' + slct_h2tr)
            
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            
            df_ind_h2 = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 6) &
                                                        (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_ind_h2))]
            df_transp_h2 = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 6) &
                                                        (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_transp_h2))]
            df_transp_el = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) &
                                                        (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_transp_el))]
            df_dmnd_ev = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & 
                                                    (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_ev))]
            # No need to scale anymore because 2050 demand as input data
            # df_transp_h2 = df_transp_h2.join(scaling_factor_h2tr, on=scaling_factor_h2tr.index.names)
            # df_transp_h2['scale'] = df_transp_h2['scale'].fillna(0) 
            # df_transp_h2['value_new'] = df_transp_h2.value * df_transp_h2.scale

            dct_dmnd_ind_h2 = (df_ind_h2.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
            dct_dmnd_transp_h2_0 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
            dct_dmnd_transp_el_0 = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
            dct_dmnd_ev_0 = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
            
            if fy_val == 2015:
                
                logger.info(msg)
                for kk, vv in dct_dmnd_ind_h2.items():
                    self.ml.m.dmnd[kk] = 0.0
                logger.info(msg_tr)
                for kk, vv in dct_dmnd_transp_h2_0.items():
                    self.ml.m.dmnd[kk] = 0.0
                logger.info(msg_tr_el)
                for kk, vv in dct_dmnd_transp_el_0.items():
                    self.ml.m.dmnd[kk] = 0.0
                logger.info(msg_tr_ev)
                for kk, vv in dct_dmnd_ev_0.items():
                    self.ml.m.dmnd[kk] = 0.0

            elif fy_val == 2020:
                
                logger.info(msg)
                for kk, vv in dct_dmnd_ind_h2.items():
                    self.ml.m.dmnd[kk] = 0.0
                logger.info(msg_tr)
                for kk, vv in dct_dmnd_transp_h2_0.items():
                    self.ml.m.dmnd[kk] = 0.0
                logger.info(msg_tr_el)
                for kk, vv in dct_dmnd_transp_el_0.items():
                    self.ml.m.dmnd[kk] = 0.0
                
                df_dmnd_ev['value_new'] = df_dmnd_ev.value * 5.830284714
                # df_transp_h2['value_new'] = df_transp_h2.value * 1/7 # If from 2020
                # df_transp_el['value_new'] = df_transp_el.value * 1/7 # If from 2020

                dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                # dct_dmnd_transp_h2 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                # dct_dmnd_transp_el = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                if slct_h2tr == '2050+':
                    
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (102.25/100)
                
                elif slct_h2tr == '100H2_all' or '100PC' in slct_h2tr:
                # elif slct_h2tr in ['100H2_all','100HCV_0LCV_100PC','100HCV_25LCV_100PC','100HCV_50LCV_100PC','100HCV_75LCV_100PC',
                #                                '75HCV_0LCV_100PC','75HCV_25LCV_100PC','75HCV_50LCV_100PC','75HCV_75LCV_100PC',
                #                                '50HCV_0LCV_100PC','50HCV_25LCV_100PC','50HCV_50LCV_100PC','50HCV_75LCV_100PC',
                #                                '25HCV_0LCV_100PC','25HCV_25LCV_100PC','25HCV_50LCV_100PC','25HCV_75LCV_100PC',
                #                                '100HCV_0LCV_100PC','100HCV_25LCV_100PC','100HCV_50LCV_100PC','100HCV_75LCV_100PC',]:       
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100EL_all' or '_0PC' in slct_h2tr:                
                # elif slct_h2tr in ['100EL_all','100HCV_0LCV_0PC','100HCV_25LCV_0PC',
                #                    '100HCV_50LCV_0PC','100HCV_75LCV_0PC','100HCV_100LCV_0PC']:
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)
                
                elif '25PC' in slct_h2tr:
                # elif slct_h2tr in ['100HCV_0LCV_25PC','100HCV_25LCV_25PC',
                #                    '100HCV_50LCV_25PC','100HCV_75LCV_25PC','100HCV_100LCV_25PC']:                
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif '50PC' in slct_h2tr:                            
                # elif slct_h2tr in ['100HCV_0LCV_50PC','100HCV_25LCV_50PC',
                #                    '100HCV_50LCV_50PC','100HCV_75LCV_50PC','100HCV_100LCV_50PC']:                   
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)  
 
                elif '75PC' in slct_h2tr:
                # elif slct_h2tr in ['100HCV_0LCV_75PC','100HCV_25LCV_75PC',
                #                    '100HCV_50LCV_75PC','100HCV_75LCV_75PC','100HCV_100LCV_75PC']:                   
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)  


            elif fy_val == 2025:
                
                logger.info(msg)
                for kk, vv in dct_dmnd_ind_h2.items():
                    self.ml.m.dmnd[kk] = 0.0
                logger.info(msg_tr)
                for kk, vv in dct_dmnd_transp_h2_0.items():
                    self.ml.m.dmnd[kk] = 0.0
                logger.info(msg_tr_el)
                for kk, vv in dct_dmnd_transp_el_0.items():
                    self.ml.m.dmnd[kk] = 0.0
                
                df_dmnd_ev['value_new'] = df_dmnd_ev.value * 45.53000509
                # df_transp_h2['value_new'] = df_transp_h2.value * 2/7 # If from 2020
                # df_transp_el['value_new'] = df_transp_el.value * 2/7 # If from 2020

                dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                # dct_dmnd_transp_h2 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                # dct_dmnd_transp_el = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                if slct_h2tr == '2050+':
                    
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                            self.ml.m.dmnd[kk] = vv * (102.25/100)
                
                elif slct_h2tr == '100H2_all' or '100PC' in slct_h2tr:
                # elif slct_h2tr in ['100H2_all','100HCV_0LCV_100PC','100HCV_25LCV_100PC','100HCV_50LCV_100PC','100HCV_75LCV_100PC',
                #                                '75HCV_0LCV_100PC','75HCV_25LCV_100PC','75HCV_50LCV_100PC','75HCV_75LCV_100PC',
                #                                '50HCV_0LCV_100PC','50HCV_25LCV_100PC','50HCV_50LCV_100PC','50HCV_75LCV_100PC',
                #                                '25HCV_0LCV_100PC','25HCV_25LCV_100PC','25HCV_50LCV_100PC','25HCV_75LCV_100PC',
                #                                '100HCV_0LCV_100PC','100HCV_25LCV_100PC','100HCV_50LCV_100PC','100HCV_75LCV_100PC',]:       
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100EL_all' or '_0PC' in slct_h2tr:                
                # elif slct_h2tr in ['100EL_all','100HCV_0LCV_0PC','100HCV_25LCV_0PC',
                #                    '100HCV_50LCV_0PC','100HCV_75LCV_0PC','100HCV_100LCV_0PC']:
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)
                
                elif '25PC' in slct_h2tr:
                # elif slct_h2tr in ['100HCV_0LCV_25PC','100HCV_25LCV_25PC',
                #                    '100HCV_50LCV_25PC','100HCV_75LCV_25PC','100HCV_100LCV_25PC']:                
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif '50PC' in slct_h2tr:                            
                # elif slct_h2tr in ['100HCV_0LCV_50PC','100HCV_25LCV_50PC',
                #                    '100HCV_50LCV_50PC','100HCV_75LCV_50PC','100HCV_100LCV_50PC']:                   
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)  
 
                elif '75PC' in slct_h2tr:
                # elif slct_h2tr in ['100HCV_0LCV_75PC','100HCV_25LCV_75PC',
                #                    '100HCV_50LCV_75PC','100HCV_75LCV_75PC','100HCV_100LCV_75PC']:                   
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)  

            elif fy_val == 2030:
                
                logger.info(msg_ind)
                for kk, vv in dct_dmnd_ind_h2.items():
                        self.ml.m.dmnd[kk] = vv
                
                df_dmnd_ev['value_new'] = df_dmnd_ev.value * 121.7429501
                # df_transp_h2['value_new'] = df_transp_h2.value * df_transp_h2.scale * 3/7 # If from 2020
                df_transp_h2['value_new'] = df_transp_h2.value * 1/5 # If from 2030
                df_transp_el['value_new'] = df_transp_el.value * 1/5 # If from 2030

                dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_h2 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_el = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                if slct_h2tr == '2050+':
                    
                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.01/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (34.58/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (102.25/100)
                        
                elif slct_h2tr == '100H2_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100EL_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (21.49/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (24.17/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.85/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (42.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (63.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (71.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (76.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (94.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (97.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (17.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (35.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (38.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (56.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (59.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (72.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (90.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (93.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (95.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (13.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (31.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (68.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (70.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (86.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (4.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (6.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (27.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (40.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (43.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (61.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (64.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (82.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (2.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (5.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (23.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (36.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (57.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.15/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.83/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (78.51/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

            elif fy_val == 2035:
                
                logger.info(msg_ind)
                for kk, vv in dct_dmnd_ind_h2.items():
                        self.ml.m.dmnd[kk] = vv
                
                df_dmnd_ev['value_new'] = df_dmnd_ev.value * 234.867546
                # df_transp_h2['value_new'] = df_transp_h2.value * df_transp_h2.scale * 3/7 # If from 2020
                df_transp_h2['value_new'] = df_transp_h2.value * 2/5 # If from 2030
                df_transp_el['value_new'] = df_transp_el.value * 2/5 # If from 2030

                dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_h2 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_el = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                if slct_h2tr == '2050+':
                    
                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.01/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (34.58/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (102.25/100)
                        
                elif slct_h2tr == '100H2_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100EL_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (21.49/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (24.17/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.85/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (42.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (63.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (71.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (76.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (94.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (97.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (17.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (35.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (38.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (56.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (59.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (72.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (90.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (93.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (95.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (13.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (31.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (68.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (70.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (86.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (4.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (6.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (27.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (40.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (43.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (61.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (64.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (82.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (2.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (5.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (23.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (36.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (57.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.15/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.83/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (78.51/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)
 
            elif fy_val == 2040:
                
                logger.info(msg_ind)
                for kk, vv in dct_dmnd_ind_h2.items():
                        self.ml.m.dmnd[kk] = vv
                
                df_dmnd_ev['value_new'] = df_dmnd_ev.value * 384.9037931
                # df_transp_h2['value_new'] = df_transp_h2.value * df_transp_h2.scale * 3/7 # If from 2020
                df_transp_h2['value_new'] = df_transp_h2.value * 3/5 # If from 2030
                df_transp_el['value_new'] = df_transp_el.value * 3/5 # If from 2030

                dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_h2 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_el = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                if slct_h2tr == '2050+':
                    
                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.01/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (34.58/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (102.25/100)
                        
                elif slct_h2tr == '100H2_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100EL_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (21.49/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (24.17/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.85/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (42.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (63.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (71.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (76.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (94.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (97.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (17.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (35.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (38.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (56.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (59.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (72.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (90.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (93.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (95.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (13.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (31.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (68.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (70.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (86.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (4.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (6.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (27.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (40.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (43.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (61.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (64.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (82.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (2.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (5.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (23.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (36.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (57.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.15/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.83/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (78.51/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

            elif fy_val == 2045:
                
                logger.info(msg_ind)
                for kk, vv in dct_dmnd_ind_h2.items():
                        self.ml.m.dmnd[kk] = vv
                
                df_dmnd_ev['value_new'] = df_dmnd_ev.value * 571.8516911
                # df_transp_h2['value_new'] = df_transp_h2.value * df_transp_h2.scale * 3/7 # If from 2020
                df_transp_h2['value_new'] = df_transp_h2.value * 4/5 # If from 2030
                df_transp_el['value_new'] = df_transp_el.value * 4/5 # If from 2030

                dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_h2 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_el = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                if slct_h2tr == '2050+':
                    
                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.01/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (34.58/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (102.25/100)
                        
                elif slct_h2tr == '100H2_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100EL_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (21.49/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (24.17/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.85/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (42.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (63.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (71.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (76.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (94.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (97.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (17.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (35.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (38.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (56.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (59.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (72.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (90.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (93.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (95.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (13.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (31.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (68.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (70.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (86.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (4.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (6.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (27.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (40.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (43.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (61.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (64.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (82.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (2.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (5.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (23.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (36.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (57.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.15/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.83/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (78.51/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)            

            elif fy_val == 2050:
                
                logger.info(msg_ind)
                for kk, vv in dct_dmnd_ind_h2.items():
                        self.ml.m.dmnd[kk] = vv
                
                df_dmnd_ev['value_new'] = df_dmnd_ev.value * 795.670272
                # df_transp_h2['value_new'] = df_transp_h2.value * df_transp_h2.scale * 3/7 # If from 2020
                df_transp_h2['value_new'] = df_transp_h2.value * 5/5 # If from 2030
                df_transp_el['value_new'] = df_transp_el.value * 5/5 # If from 2030

                dct_dmnd_ev = (df_dmnd_ev.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_h2 = (df_transp_h2.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                dct_dmnd_transp_el = (df_transp_el.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                if slct_h2tr == '2050+':
                    
                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.01/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (34.58/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (102.25/100)
                        
                elif slct_h2tr == '100H2_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100EL_all':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (21.49/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (24.17/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.85/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '100HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (42.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '100HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (63.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '100HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (71.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (76.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (0/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '100HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (40.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (30.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (94.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (20.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '100HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (97.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (10.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (17.46/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.14/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.82/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '75HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (35.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (38.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '75HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (56.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (59.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '75HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (72.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '75HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (55.03/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (45.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (90.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (35.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (93.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '75HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (95.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (14.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (13.43/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (16.10/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.78/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '50HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (31.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (34.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (37.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '50HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (52.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (55.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '50HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (68.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (70.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '50HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (70.02/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (60.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (86.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (50.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (89.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (39.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '50HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (91.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (29.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_0LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (4.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (6.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (9.39/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (12.07/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (14.75/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '25HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (22.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (25.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (27.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (30.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (33.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '25HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (40.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (43.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (45.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (48.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (51.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '25HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (58.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (61.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (64.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (66.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (69.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '25HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (77.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (85.01/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (79.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (75.00/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (82.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (64.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (85.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (54.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '25HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (87.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (44.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (2.68/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_50LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (5.36/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_75LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (8.03/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_100LCV_0PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (10.71/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (100/100)

                elif slct_h2tr == '0HCV_0LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (18.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_25LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (20.97/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_50LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (23.64/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_75LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (26.32/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_100LCV_25PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (29.00/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (75/100)

                elif slct_h2tr == '0HCV_0LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (36.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_25LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (39.25/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_50LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (41.93/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_75LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (44.61/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_100LCV_50PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (47.29/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (50/100)

                elif slct_h2tr == '0HCV_0LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (54.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_25LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (57.54/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_50LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (60.22/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_75LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (62.90/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_100LCV_75PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (65.57/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (25/100)

                elif slct_h2tr == '0HCV_0LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (73.15/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (100/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_25LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (75.83/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (89.99/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_50LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (78.51/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (79.98/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_75LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (81.18/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (69.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)

                elif slct_h2tr == '0HCV_100LCV_100PC':

                    logger.info(msg_transp)
                    for kk, vv in dct_dmnd_transp_h2.items():
                        self.ml.m.dmnd[kk] = vv * (83.86/100)
                    logger.info(msg_transp_el)
                    for kk, vv in dct_dmnd_transp_el.items():
                        self.ml.m.dmnd[kk] = vv * (59.97/100) 
                    logger.info(msg_ev)
                    for kk, vv in dct_dmnd_ev.items():
                        self.ml.m.dmnd[kk] = vv * (0/100)


            #  Decrease DHW boilers demand when full of fossil
            
            df_dhw_bo = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id == 3]
            df_dhw_bo = df_dhw_bo.join(scaling_factor, on=scaling_factor.index.names)
            df_dhw_bo['scale'] = df_dhw_bo['scale'].fillna(0) 
            df_dhw_bo['value_new'] = df_dhw_bo.value * df_dhw_bo.scale
            
            dct_dmnd_dhw_bo = (df_dhw_bo.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
            
            if slct_hp == 'bau':
                
                dmnd_dhw_2050_pct = 1
                msg_dhw_bo = ('Setting DHW demand supply by boilers for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(dmnd_dhw_2050_pct * 100) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                logger.info(msg_dhw_bo)
                for kk, vv in dct_dmnd_dhw_bo.items():
                    self.ml.m.dmnd[kk] = vv

            if slct_hp == 'fossil':
                
                dmnd_dhw_2050_pct = 0.448
                msg_dhw_bo = ('Setting DHW demand supply by boilers for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(dmnd_dhw_2050_pct * 100) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)
                        
            if slct_hp == 'full':

                dmnd_dhw_2050_pct = 0.197
                msg_dhw_bo = ('Setting DHW demand supply by boilers for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(dmnd_dhw_2050_pct * 100) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_dmnd_dhw_bo.items():
                        self.ml.m.dmnd[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)
                        
            
                          
            # HEAT
            pf_ca_id_1 = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 1]['dmnd_pf_id'].to_list()
            pf_ca_id_2 = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 2]['dmnd_pf_id'].to_list()
            dct_pf_id_dmnd_ht = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin([1,2])].set_index('dmnd_pf_id')['nd_id'].to_dict()
            list_pf_id_dmnd_ht = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin([1,2])]['dmnd_pf_id'].to_list()
            dict_tm_soy = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_ht][['sy','doy']].set_index('doy')['sy'].to_dict()
            
            df_ht_other = self.ml.m.df_profdmnd.loc[self.ml.m.df_profdmnd.dmnd_pf_id.isin(list_pf_id_dmnd_ht)].copy()
            df_ht_other['nd_id'] = df_ht_other['dmnd_pf_id'].map(dct_pf_id_dmnd_ht)
            df_ht_other['sy'] = df_ht_other['doy'].map(dict_tm_soy)
            df_ht_other['ca_id'] = 0
            df_ht_other.loc[df_ht_other.dmnd_pf_id.isin(pf_ca_id_1),'ca_id'] = 1
            df_ht_other.loc[df_ht_other.dmnd_pf_id.isin(pf_ca_id_2),'ca_id'] = 2

                
            df_ht_other = df_ht_other.drop(columns=['dmnd_pf_id', 'hy', 'value','doy']).join(scaling_factor_ht, on=scaling_factor_ht.index.names)
            
            # DHW AW and WW
            pf_ca_id_4 = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 4]['dmnd_pf_id'].to_list()
            pf_ca_id_5 = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id == 5]['dmnd_pf_id'].to_list()
            dct_pf_id_dmnd_dhw = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin([4,5])].set_index('dmnd_pf_id')['nd_id'].to_dict()
            list_pf_id_dmnd_dhw = self.ml.m.df_node_encar.loc[self.ml.m.df_node_encar.ca_id.isin([4,5])]['dmnd_pf_id'].to_list()
            dict_tm_soy = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_ht][['sy','doy']].set_index('doy')['sy'].to_dict()
            
            df_dhw_other = self.ml.m.df_profdmnd.loc[self.ml.m.df_profdmnd.dmnd_pf_id.isin(list_pf_id_dmnd_dhw)].copy()
            df_dhw_other['nd_id'] = df_dhw_other['dmnd_pf_id'].map(dct_pf_id_dmnd_dhw)
            df_dhw_other['sy'] = df_dhw_other['doy'].map(dict_tm_soy)
            df_dhw_other['ca_id'] = 0
            df_dhw_other.loc[df_dhw_other.dmnd_pf_id.isin(pf_ca_id_4),'ca_id'] = 4
            df_dhw_other.loc[df_dhw_other.dmnd_pf_id.isin(pf_ca_id_5),'ca_id'] = 5

            df_dhw_other = df_dhw_other.drop(columns=['dmnd_pf_id', 'hy', 'value','doy']).join(scaling_factor_dhw, on=scaling_factor_dhw.index.names)           
            
            if slct_rf == 'default':
                
                msg = ('Setting special heating dmnd_sum to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col) + ' for retrofit scenario ' + slct_rf + ' and for HP scenario ' + slct_hp)

                logger.info(msg)
                #HEAT
                df_ht = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id.isin([1,2])]
                df_ht = df_ht.join(scaling_factor_ht, on=scaling_factor_ht.index.names)
                df_ht['value_new'] = df_ht.value * df_ht.scale
    
                dct_dmnd_ht = (df_ht.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                df_ht_fossil = df_ht_other.copy()
                df_ht_fossil['value_new'] = df_ht_fossil.erg_tot_fossil * df_ht_fossil.scale
                
                df_ht_fossil = df_ht_fossil.loc[~df_ht_fossil.sy.isna()]
                dct_dmnd_ht_fossil = (df_ht_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                # DHW for AW and WW
                df_dhw = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id.isin([4,5])]
                df_dhw = df_dhw.join(scaling_factor_dhw, on=scaling_factor_dhw.index.names)
                df_dhw['value_new'] = df_dhw.value * df_dhw.scale
    
                dct_dmnd_dhw = (df_dhw.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                df_dhw_fossil = df_dhw_other.copy()
                df_dhw_fossil['value_new'] = df_dhw_fossil.erg_tot_fossil * df_dhw_fossil.scale
                
                df_dhw_fossil = df_dhw_fossil.loc[~df_dhw_fossil.sy.isna()]
                dct_dmnd_dhw_fossil = (df_dhw_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                if slct_hp == 'full':
               
                    for kk, vv in dct_dmnd_ht.items():
                        self.ml.m.dmnd[kk] = vv
                    
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv
                        
                elif slct_hp == 'bau':
                  
                    for kk, vv in dct_dmnd_ht.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                        
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv * 0.568 
                        
                elif slct_hp == 'fossil':
                    
                    for kk, vv in dct_dmnd_ht_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                    
                    for kk, vv in dct_dmnd_dhw_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                        
            elif slct_rf == 'retr_1pc':
                
                msg = ('Setting special heating dmnd_sum to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col) + ' for retrofit scenrio ' + slct_rf + ' and for HP scenario ' + slct_hp)

                logger.info(msg)
                
                # HEAT
                df_ht_retr_1pc = self.ml.m.df_profdmnd.loc[self.ml.m.df_profdmnd.dmnd_pf_id.isin(list_pf_id_dmnd_ht)].copy()
                df_ht_retr_1pc['nd_id'] = df_ht_retr_1pc['dmnd_pf_id'].map(dct_pf_id_dmnd_ht)
                df_ht_retr_1pc['sy'] = df_ht_retr_1pc['doy'].map(dict_tm_soy)
                df_ht_retr_1pc['ca_id'] = 0
                df_ht_retr_1pc.loc[df_ht_retr_1pc.dmnd_pf_id.isin(pf_ca_id_1),'ca_id'] = 1
                df_ht_retr_1pc.loc[df_ht_retr_1pc.dmnd_pf_id.isin(pf_ca_id_2),'ca_id'] = 2

                
                df_ht_retr_1pc = df_ht_retr_1pc[['nd_id', 'sy','erg_tot_retr_1pc','ca_id']].join(scaling_factor_ht, on=scaling_factor_ht.index.names)
                df_ht_retr_1pc['value_new'] = df_ht_retr_1pc.erg_tot_retr_1pc * df_ht_retr_1pc.scale
                
                df_ht_retr_1pc = df_ht_retr_1pc.loc[~df_ht_retr_1pc.sy.isna()]
                dct_dmnd_ht_retr_1pc = (df_ht_retr_1pc.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                
                df_ht_retr_1pc_fossil = df_ht_other.copy()
                df_ht_retr_1pc_fossil['value_new'] = df_ht_retr_1pc_fossil.erg_tot_retr_1pc_fossil * df_ht_retr_1pc_fossil.scale
                
                df_ht_retr_1pc_fossil = df_ht_retr_1pc_fossil.loc[~df_ht_retr_1pc_fossil.sy.isna()]
                dct_dmnd_ht_retr_1pc_fossil = (df_ht_retr_1pc_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                # DHW for AW and WW
                df_dhw = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id.isin([4,5])]
                df_dhw = df_dhw.join(scaling_factor_dhw, on=scaling_factor_dhw.index.names)
                df_dhw['value_new'] = df_dhw.value * df_dhw.scale
    
                dct_dmnd_dhw = (df_dhw.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                df_dhw_fossil = df_dhw_other.copy()
                df_dhw_fossil['value_new'] = df_dhw_fossil.erg_tot_fossil * df_dhw_fossil.scale
                
                df_dhw_fossil = df_dhw_fossil.loc[~df_dhw_fossil.sy.isna()]
                dct_dmnd_dhw_fossil = (df_dhw_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                if slct_hp == 'full':
                 
                    for kk, vv in dct_dmnd_ht_retr_1pc.items():
                        self.ml.m.dmnd[kk] = vv
                    
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv
                        
                elif slct_hp == 'bau':
                   
                    for kk, vv in dct_dmnd_ht_retr_1pc.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                    
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil':
                    
                    for kk, vv in dct_dmnd_ht_retr_1pc_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                        
                    for kk, vv in dct_dmnd_dhw_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                        
            elif slct_rf == 'retr_2pc':
                
                msg = ('Setting special heating dmnd_sum to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col) + ' for retrofit scenario ' + slct_rf+ ' and for HP scenario ' + slct_hp)

                logger.info(msg)
                
                #HEAT
                df_ht_retr_2pc = self.ml.m.df_profdmnd.loc[self.ml.m.df_profdmnd.dmnd_pf_id.isin(list_pf_id_dmnd_ht)].copy()
                df_ht_retr_2pc['nd_id'] = df_ht_retr_2pc['dmnd_pf_id'].map(dct_pf_id_dmnd_ht)
                df_ht_retr_2pc['sy'] = df_ht_retr_2pc['doy'].map(dict_tm_soy)
                df_ht_retr_2pc['ca_id'] = 0
                df_ht_retr_2pc.loc[df_ht_retr_2pc.dmnd_pf_id.isin(pf_ca_id_1),'ca_id'] = 1
                df_ht_retr_2pc.loc[df_ht_retr_2pc.dmnd_pf_id.isin(pf_ca_id_2),'ca_id'] = 2

                
                df_ht_retr_2pc = df_ht_retr_2pc[['nd_id', 'sy','erg_tot_retr_2pc','ca_id']].join(scaling_factor_ht, on=scaling_factor_ht.index.names)
                df_ht_retr_2pc['value_new'] = df_ht_retr_2pc.erg_tot_retr_2pc * df_ht_retr_2pc.scale
                
                df_ht_retr_2pc = df_ht_retr_2pc.loc[~df_ht_retr_2pc.sy.isna()]
                dct_dmnd_ht_retr_2pc = (df_ht_retr_2pc.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                df_ht_retr_2pc_fossil = df_ht_other.copy()
                df_ht_retr_2pc_fossil['value_new'] = df_ht_retr_2pc_fossil.erg_tot_retr_2pc_fossil * df_ht_retr_2pc_fossil.scale
                
                df_ht_retr_2pc_fossil = df_ht_retr_2pc_fossil.loc[~df_ht_retr_2pc_fossil.sy.isna()]
                dct_dmnd_ht_retr_2pc_fossil = (df_ht_retr_2pc_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                # DHW for AW and WW
                df_dhw = IO.param_to_df(self.ml.m.dmnd).loc[IO.param_to_df(self.ml.m.dmnd).ca_id.isin([4,5])]
                df_dhw = df_dhw.join(scaling_factor_dhw, on=scaling_factor_dhw.index.names)
                df_dhw['value_new'] = df_dhw.value * df_dhw.scale
    
                dct_dmnd_dhw = (df_dhw.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                df_dhw_fossil = df_dhw_other.copy()
                df_dhw_fossil['value_new'] = df_dhw_fossil.erg_tot_fossil * df_dhw_fossil.scale
                
                df_dhw_fossil = df_dhw_fossil.loc[~df_dhw_fossil.sy.isna()]
                dct_dmnd_dhw_fossil = (df_dhw_fossil.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                
                if slct_hp == 'full':
                 
                    for kk, vv in dct_dmnd_ht_retr_2pc.items():
                        self.ml.m.dmnd[kk] = vv
                        
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv
                        
                elif slct_hp == 'bau':
                   
                    for kk, vv in dct_dmnd_ht_retr_2pc.items():
                        self.ml.m.dmnd[kk] = vv * 0.568
                    
                    for kk, vv in dct_dmnd_dhw.items():
                        self.ml.m.dmnd[kk] = vv * 0.568 
                        
                elif slct_hp == 'fossil':
                    
                    for kk, vv in dct_dmnd_ht_retr_2pc_fossil.items():
                        self.ml.m.dmnd[kk] = vv
                    
                    for kk, vv in dct_dmnd_dhw_fossil.items():
                        self.ml.m.dmnd[kk] = vv
            
            
            
            if slct_dpf == 'ee': # Energy efficiency
            
                slct_col = 'dmnd_sum' + str_fy
                slct_col_prev = 'dmnd_sum' + str_fyp
                
                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]
                
                msg = ('Setting energy efficiency curves to residential and DSR for ' + str_fy.replace('_', ' ')
                      + ' from column {}'.format(slct_col) + ' for ee scenario ' + slct_dpf)
    
                logger.info(msg)
                
                # Load before as we need both for EE and DSR
                path_prof_id = os.path.join(config.PATH_CSV, 'def_profile.csv')
                path_dmnd = os.path.join(config.PATH_CSV, 'profdmnd.csv')
                
                df_def_profile = pd.read_csv(path_prof_id)
                df_dmnd = pd.read_csv(path_dmnd)[['dmnd_pf_id', 'hy', 'value', 'doy']]

                # Energy efficiency
                
                list_nd_name_res_el = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].nd.to_list()
                dct_nd_id_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].set_index('nd')['nd_id'].to_dict()

                df_def_profile_ee = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('diff'))&
                                                              (df_def_profile.primary_nd.isin(list_nd_name_res_el))&
                                                              ~(df_def_profile.pf.str.contains('best'))]
                df_def_profile_ee['nd_id'] = df_def_profile_ee['primary_nd'].map(dct_nd_id_res)
                df_def_profile_ee_2035 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2035_2015')]
                df_def_profile_ee_2050 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2050_2015')]
                
                df_dmnd_ee_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2035 = pd.merge(df_dmnd_ee_2035,df_def_profile_ee_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_ee_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2050 = pd.merge(df_dmnd_ee_2050,df_def_profile_ee_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_el][['sy','hy']].set_index('hy')['sy'].to_dict()
                df_dmnd_ee_2035['sy'] = df_dmnd_ee_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_ee_2050['sy'] = df_dmnd_ee_2050['hy'].map(dict_tm_soy_hour)
                df_el_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res))]
                
                # DSR load under energy efficiency
                
                list_nd_name_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].nd.to_list()
                dct_nd_id_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].set_index('nd')['nd_id'].to_dict()
    
                df_def_profile_dsr = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('DSR'))&
                                                               (df_def_profile.primary_nd.isin(list_nd_name_dsr))&
                                                              ~(df_def_profile.pf.str.contains('best'))]
                df_def_profile_dsr['nd_id'] = df_def_profile_dsr['primary_nd'].map(dct_nd_id_dsr)
                # df_def_profile_dsr_2015 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2015')]
                df_def_profile_dsr_2035 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2035')]
                df_def_profile_dsr_2050 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2050')]
                
                # df_dmnd_dsr_2015 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2015.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                # df_dmnd_dsr_2015 = pd.merge(df_dmnd_dsr_2015,df_def_profile_dsr_2015,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2035 = pd.merge(df_dmnd_dsr_2035,df_def_profile_dsr_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2050 = pd.merge(df_dmnd_dsr_2050,df_def_profile_dsr_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_dsr][['sy','hy']].set_index('hy')['sy'].to_dict()
                # df_dmnd_dsr_2015['sy'] = df_dmnd_dsr_2015['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2035['sy'] = df_dmnd_dsr_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2050['sy'] = df_dmnd_dsr_2050['hy'].map(dict_tm_soy_hour)
                df_el_dsr = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_dsr))]

                if fy_val == 2015:
                    
                    dct_dmnd_res = (df_el_res.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_res.items():
                        self.ml.m.dmnd[kk] = vv
                    dct_dmnd_dsr = (df_el_dsr.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_dsr.items():
                        self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(1/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(1/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                   
                elif fy_val == 2025:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(2/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(2/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(3/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(3/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(4/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(4/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(5/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(5/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(6/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(6/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(7/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(7/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                        
            
            elif slct_dpf == 'best': # BEST Energy efficiency
            
                slct_col = 'dmnd_sum' + str_fy
                slct_col_prev = 'dmnd_sum' + str_fyp
                
                dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
                slct_fy = self.ml.dct_step['swfy']
                fy_val = dict_fy[slct_fy]
                
                msg = ('Setting energy efficiency curves to residential and DSR for ' + str_fy.replace('_', ' ')
                      + ' from column {}'.format(slct_col) + ' for ee scenario ' + slct_dpf)
    
                logger.info(msg)
                    
                # Load before as we need both for EE and DSR
                path_prof_id = os.path.join(config.PATH_CSV, 'def_profile.csv')
                path_dmnd = os.path.join(config.PATH_CSV, 'profdmnd.csv')
                
                df_def_profile = pd.read_csv(path_prof_id)
                df_dmnd = pd.read_csv(path_dmnd)[['dmnd_pf_id', 'hy', 'value', 'doy']]
                # Energy efficiency
                
                list_nd_name_res_el = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].nd.to_list()
                dct_nd_id_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_el_res)].set_index('nd')['nd_id'].to_dict()

                df_def_profile_ee = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('best_2035|2050'))&(df_def_profile.primary_nd.isin(list_nd_name_res_el))]
                df_def_profile_ee['nd_id'] = df_def_profile_ee['primary_nd'].map(dct_nd_id_res)
                df_def_profile_ee_2035 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2035_2015')]
                df_def_profile_ee_2050 = df_def_profile_ee.loc[df_def_profile_ee.pf.str.contains('2050_2015')]
                
                df_dmnd_ee_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2035 = pd.merge(df_dmnd_ee_2035,df_def_profile_ee_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_ee_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_ee_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'diff_erg'})
                df_dmnd_ee_2050 = pd.merge(df_dmnd_ee_2050,df_def_profile_ee_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_el][['sy','hy']].set_index('hy')['sy'].to_dict()
                df_dmnd_ee_2035['sy'] = df_dmnd_ee_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_ee_2050['sy'] = df_dmnd_ee_2050['hy'].map(dict_tm_soy_hour)
                # df_dmnd_ee_2035 = df_dmnd_ee_2035.loc[~df_dmnd_ee_2035]
                df_el_res = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_el_res))]

                # DSR load under energy efficiency
                
                list_nd_name_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].nd.to_list()
                dct_nd_id_dsr = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd_id.isin(nd_arch_dsr)].set_index('nd')['nd_id'].to_dict()
    
                df_def_profile_dsr = df_def_profile.copy().loc[(df_def_profile.pf.str.contains('DSR_best_2035|DSR_2050'))&
                                                               (df_def_profile.primary_nd.isin(list_nd_name_dsr))]
                df_def_profile_dsr['nd_id'] = df_def_profile_dsr['primary_nd'].map(dct_nd_id_dsr)
                # df_def_profile_dsr_2015 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2015')]
                df_def_profile_dsr_2035 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2035')]
                df_def_profile_dsr_2050 = df_def_profile_dsr.loc[df_def_profile_dsr.pf.str.contains('2050')]
                
                # df_dmnd_dsr_2015 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2015.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                # df_dmnd_dsr_2015 = pd.merge(df_dmnd_dsr_2015,df_def_profile_dsr_2015,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2035 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2035.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2035 = pd.merge(df_dmnd_dsr_2035,df_def_profile_dsr_2035,left_on='dmnd_pf_id', right_on='pf_id')
                df_dmnd_dsr_2050 = df_dmnd.copy().loc[df_dmnd.dmnd_pf_id.isin(df_def_profile_dsr_2050.pf_id.to_list())].reset_index(drop=True).rename(columns={'value':'dsr_load'})
                df_dmnd_dsr_2050 = pd.merge(df_dmnd_dsr_2050,df_def_profile_dsr_2050,left_on='dmnd_pf_id', right_on='pf_id')
                
                dict_tm_soy_hour = self.ml.m.df_tm_soy_full.loc[self.ml.m.df_tm_soy_full.tm_id==tm_id_dsr][['sy','hy']].set_index('hy')['sy'].to_dict()
                # df_dmnd_dsr_2015['sy'] = df_dmnd_dsr_2015['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2035['sy'] = df_dmnd_dsr_2035['hy'].map(dict_tm_soy_hour)
                df_dmnd_dsr_2050['sy'] = df_dmnd_dsr_2050['hy'].map(dict_tm_soy_hour)
                df_el_dsr = IO.param_to_df(self.ml.m.dmnd).loc[(IO.param_to_df(self.ml.m.dmnd).ca_id == 0) & (IO.param_to_df(self.ml.m.dmnd).nd_id.isin(nd_arch_dsr))]
                 
                if fy_val == 2015:
                    
                    dct_dmnd_res = (df_el_res.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_res.items():
                        self.ml.m.dmnd[kk] = vv
                    dct_dmnd_dsr = (df_el_dsr.set_index(['sy', 'nd_id', 'ca_id'])['value'].to_dict())
                    for kk, vv in dct_dmnd_dsr.items():
                        self.ml.m.dmnd[kk] = vv
                    
                elif fy_val == 2020:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(1/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(1/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                   
                elif fy_val == 2025:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(2/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(2/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv

                
                elif fy_val == 2030:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(3/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(3/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2035:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(4/4))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2035,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(4/4))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2040:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(5/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(5/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2045:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(6/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(6/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
                elif fy_val == 2050:
                    
                    df_el_res_ee = pd.merge(df_el_res,df_dmnd_ee_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value + x.diff_erg*(7/7))
                    df_el_dsr_ee = pd.merge(df_el_dsr,df_dmnd_dsr_2050,on=['sy','nd_id']).assign(
                                            value_new = lambda x: x.value - (x.value - x.dsr_load)*(7/7))
                    
                    dct_dmnd_res_ee = (df_el_res_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())
                    dct_dmnd_dsr_ee = (df_el_dsr_ee.set_index(['sy', 'nd_id', 'ca_id'])['value_new'].to_dict())

                    for kk, vv in dct_dmnd_res_ee.items():
                        self.ml.m.dmnd[kk] = vv
                    for kk, vv in dct_dmnd_dsr_ee.items():
                        self.ml.m.dmnd[kk] = vv
                
        #######################################################################
        def set_co2_price(str_fy=None):
            ''' Select CO2 price for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'price_co2' + str_fy
            msg = ('Setting price_co2  to values' + str_fy.replace('_', ' ')
                   + ' from column {}'.format(slct_col))
            logger.info(msg)

            df_new = self.ml.m.df_def_node[['nd_id', slct_col]]
            par = self.ml.m.dict_par['price_co2']
            col_mt_fact = 'mt_fact' if not str_fy else 'mt_fact_others'
            par.init_update(df_new, col_mt_fact)

        #######################################################################

        def set_erg_inp(slct_ch, str_fy=None, excl_fl=[]):
            ''' Select exogenous energy production for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'erg_inp' + str_fy
            msg = ('Setting erg_inp to values ' + str_fy.replace('_', '')
                   + 'from column {}.'.format(slct_col))
            logger.info(msg)

            ind_col = ['nd_id', 'ca_id', 'fl_id']
            mask_excl = -self.ml.m.df_fuel_node_encar.fl_id.isin(excl_fl)
            dct_erg_inp = (self.ml.m.df_fuel_node_encar.loc[mask_excl]
                               .set_index(ind_col)[slct_col].to_dict())

            if slct_ch != 'default':

                df = self.ml.m.df_fuel_node_encar_scenarios.copy()
                df = df.loc[df.scenario == slct_ch]
                dct_erg_inp_scen = (df.set_index(ind_col)[slct_col].to_dict())
                dct_erg_inp.update(dct_erg_inp_scen)

            for kk, vv in dct_erg_inp.items():
                self.ml.m.erg_inp[kk] = vv
        #######################################################################

        def set_erg_chp(str_fy=None, excl_pp=[]):
            ''' Select exogenous chp energy production for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'erg_chp' +  str_fy
            msg = ('Setting erg_chp to values ' + str_fy.replace('_', '')
                   + ' from column {}.'.format(slct_col))
            logger.info(msg)

            dct_erg_chp = (self.ml.m.df_plant_encar
                               .loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
                                   & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.chp)]
                               .set_index(['pp_id', 'ca_id'])[slct_col]
                               .to_dict())

            for kk, vv in dct_erg_chp.items():
                self.ml.m.erg_chp[kk] = vv

        #######################################################################
        
        def set_fc_cap(str_fy=None, excl_pp=[]):
            ''' Select capital costs for power plants for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'fc_cp_ann' +  str_fy
            msg = ('Setting fc_cp_ann to values ' + str_fy.replace('_', '')
                   + ' from column {}.'.format(slct_col))
            logger.info(msg)

            dct_fc_cp_ann = (self.ml.m.df_plant_encar
                               .loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
                                   & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.add)]
                               .set_index(['pp_id', 'ca_id'])[slct_col]
                               .to_dict())

            for kk, vv in dct_fc_cp_ann.items():
                self.ml.m.fc_cp_ann[kk] = vv
            
        
#           To chekc if it is working (*0.0001)    
            # list_pp_fc =  self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('FC')].pp_id.tolist()       
            # list_pp_ely_fc =  self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('ELY|FC')].pp_id.tolist()       

            # dct_fc_cp_ann_sto = (self.ml.m.df_plant_encar
            #                   .loc[self.ml.m.df_plant_encar.pp_id.isin(list_pp_ely_fc)
            #                       # & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.add)
            #                       ]
            #                   .set_index(['pp_id', 'ca_id'])[slct_col]
            #                   .to_dict())

            # for kk, vv in dct_fc_cp_ann_sto.items():
            #     self.ml.m.fc_cp_ann[kk] = vv * 0.0001
        #######################################################################
        
        def set_fc_om(str_fy=None, excl_pp=[]):
            ''' Select O&M costs for power plants for selected year. '''

            if str_fy == None:
                str_fy = ''

            slct_col = 'fc_om' +  str_fy
            msg = ('Setting fc_om to values ' + str_fy.replace('_', '')
                   + ' from column {}.'.format(slct_col))
            logger.info(msg)

            dct_fc_om = (self.ml.m.df_plant_encar
                               .loc[-self.ml.m.df_plant_encar.pp_id.isin(excl_pp)
                                   & self.ml.m.df_plant_encar.pp_id.isin(self.ml.m.add)]
                               .set_index(['pp_id', 'ca_id'])[slct_col]
                               .to_dict())

            for kk, vv in dct_fc_om.items():
                self.ml.m.fc_om[kk] = vv
        
        #######################################################################
        def set_dhw_capacity(slct_hp, str_fy=None):
            ''' Decrease DHW boilers capacity when full of fossil'''
            
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            list_dhw_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_STO')].pp_id.tolist()
            list_dhw_bo = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('DHW_BO')].pp_id.tolist()

            # Decrease capacity of the boilers 
            dct_capdhwbosto = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_bo + list_dhw_sto)]
                                 .set_index(['pp_id', 'ca_id'])['cap_pwr_leg']
                                 .to_dict())

            if slct_hp == 'bau':
                
                dmnd_dhw_2050_pct = 1
                msg_dhw_bo = ('Setting DHW boilers capacity for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(dmnd_dhw_2050_pct * 100) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                logger.info(msg_dhw_bo)
                for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
               
            if slct_hp == 'fossil':
                
                dmnd_dhw_2050_pct = 0.448
                msg_dhw_bo = ('Setting DHW boilers capacity for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(dmnd_dhw_2050_pct * 100) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)
                        
            if slct_hp == 'full':

                dmnd_dhw_2050_pct = 0.197
                msg_dhw_bo = ('Setting DHW boilers capacity for' + str_fy.replace('_', ' ')
                   +  ' to values ' + str(dmnd_dhw_2050_pct * 100) + '% in 2050 ' + 'for HP scenario ' + slct_hp)

                if fy_val == 2015:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                elif fy_val == 2020:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*1/7)
                elif fy_val == 2025:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*2/7)
                elif fy_val == 2030:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*3/7)
                elif fy_val == 2035:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*4/7)
                elif fy_val == 2040:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*5/7)
                elif fy_val == 2045:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*6/7)
                elif fy_val == 2050:
                    logger.info(msg_dhw_bo)
                    for kk, vv in dct_capdhwbosto.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * (1-(1-dmnd_dhw_2050_pct)*7/7)

        #######################################################################
                        
        def set_hp_capacity(slct_hp, slct_rf, str_fy=None):
            ''' Select heat pump scenario for archetypes'''
            
            if str_fy == None:
                str_fy = ''
                
            if slct_rf == 'default':
                
                slct_col = 'cap_pwr_leg' + str_fy
                slct_col_fossil = 'cap_pwr_leg' + str_fy + '_hp_' + slct_hp
                
                msg = ('Setting cap_pwr_leg to values for HP' + str_fy.replace('_', ' ') +
                       ' from column {}'.format(slct_col) + ' for scenario ' + slct_hp + ' and retrofit ' + slct_rf)
                logger.info(msg)
                
                dct_caphp = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                dct_capdhw = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                
                if slct_hp == 'full': # full heat pumps in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                 
                elif slct_hp == 'bau': # bau heat pumps deployement in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil': # fossil fuel replacement heat pumps deployement in 2050
                    
                    dct_caphp_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                
                    dct_capdhw_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                    
                    for kk, vv in dct_caphp_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
            
            elif slct_rf == 'retr_1pc':
                
                slct_col = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf
                slct_col_fossil = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf + '_' + slct_hp

                msg = ('Setting cap_pwr_leg to values for HP' + str_fy.replace('_', ' ') +
                       ' from column {}'.format(slct_col) + ' for scenario ' + slct_hp + ' and retrofit ' + slct_rf)
                logger.info(msg)
                
                dct_caphp = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                dct_capdhw = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                
                if slct_hp == 'full': # full heat pumps in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                 
                elif slct_hp == 'bau': # bau heat pumps deployement in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil': # fossil fuel replacement heat pumps deployement in 2050
                    
                    dct_caphp_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                
                    dct_capdhw_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                    
                    for kk, vv in dct_caphp_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv

           
            elif slct_rf == 'retr_2pc':
                
                slct_col = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf
                slct_col_fossil = 'cap_pwr_leg' + str_fy + '_hp_' + slct_rf + '_' + slct_hp

                msg = ('Setting cap_pwr_leg to values for HP' + str_fy.replace('_', ' ') +
                       ' from column {}'.format(slct_col) + ' for scenario ' + slct_hp + ' and retrofit ' + slct_rf)
                logger.info(msg)
                
                dct_caphp = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                dct_capdhw = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col]
                                 .to_dict())
                
                
                if slct_hp == 'full': # full heat pumps in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                 
                elif slct_hp == 'bau': # bau heat pumps deployement in 2050
                    
                    for kk, vv in dct_caphp.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                    for kk, vv in dct_capdhw.items():
                        self.ml.m.cap_pwr_leg[kk] = vv * 0.568
                        
                elif slct_hp == 'fossil': # fossil fuel replacement heat pumps deployement in 2050
                    
                    dct_caphp_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_hp_aw + list_hp_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                
                    dct_capdhw_fossil = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_dhw_aw + list_dhw_ww)]
                                 .set_index(['pp_id', 'ca_id'])[slct_col_fossil]
                                 .to_dict())
                    
                    for kk, vv in dct_caphp_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv
                        
                    for kk, vv in dct_capdhw_fossil.items():
                        self.ml.m.cap_pwr_leg[kk] = vv

        #######################################################################
                      
        def set_cap_trm_leg(slct_tr, str_fy=None):
            ''' Select transmission capacity scenario for archetypes. '''
            
            # nd_arch = self.ml.m.df_def_node.loc[np.invert(self.ml.m.df_def_node.nd.isin(['CH0', 'AT0', 'IT0', 'FR0', 'DE0']))].nd_id.tolist()
            # nd_arch_ht = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('HT')].nd_id.tolist()
            # nd_arch_res = self.ml.m.df_def_node.loc[self.ml.m.df_def_node.nd.str.contains('SFH|MFH')].nd_id.tolist()
            # nd_arch_el = list(set(nd_arch) - set(nd_arch_ht))
            # nd_arch_el_res = list(set(nd_arch_res) - set(nd_arch_ht))
            # nd_arch_el_non_res = list(set(nd_arch_el) - set(nd_arch_el_res))
            
            if str_fy == None:
                str_fy = ''

            slct_col_trme = 'cap_trme_leg' 
            slct_col_trmi = 'cap_trmi_leg' #+ (str_hy if not reset else '')
            msg = ('Setting transmission capacity for archetypes'+ str_fy.replace('_', ' ') +
                    ' for scenario ' + slct_tr)
            logger.info(msg)
            
            dct_cap_trme_leg = (self.ml.m.df_node_connect
                                   .loc[self.ml.m.df_node_connect.nd_2_id.isin(nd_arch_el)]
                                   .set_index(['mt_id', 'nd_id',
                                               'nd_2_id', 'ca_id'])[slct_col_trme]
                                   .to_dict())
            dct_cap_trmi_leg = (self.ml.m.df_node_connect
                                   .loc[self.ml.m.df_node_connect.nd_2_id.isin(nd_arch_el)]
                                   .set_index(['mt_id', 'nd_id',
                                               'nd_2_id', 'ca_id'])[slct_col_trmi]
                                   .to_dict())
            
            slct_fy = self.ml.dct_step['swfy']
            fy_val = dict_fy[slct_fy]
            
            if fy_val == 2015:
                
                for kk, vv in dct_cap_trme_leg.items():
                    self.ml.m.cap_trme_leg[kk] = vv
                for kk, vv in dct_cap_trmi_leg.items():
                    self.ml.m.cap_trmi_leg[kk] = vv
                    
            else:
                
                if slct_tr == 'default':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv
                
                elif slct_tr == 'min_5pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*0.95
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*0.95
                
                elif slct_tr == 'plus_5pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*1.05
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*1.05
                
                elif slct_tr == 'plus_10pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*1.1
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*1.1
                        
                elif slct_tr == 'plus_15pc':
                    
                    for kk, vv in dct_cap_trme_leg.items():
                        self.ml.m.cap_trme_leg[kk] = vv*1.15
                    for kk, vv in dct_cap_trmi_leg.items():
                        self.ml.m.cap_trmi_leg[kk] = vv*1.15
                                    
        #######################################################################
#
#        def set_supprof(str_hy=None, reset=False):
#            '''
#            Select supply profiles for selected year.
#            (Normalized) input profiles are scaled by erg_inp
#            and then divided by cap_pwr_leg to get the hourly capacity
#            factor.
#            '''
#
#            if str_hy == None:
#                reset=True
#                str_hy = ''
#
#            msg = ('Resetting supprof to base year values.'
#                   if reset else
#                   'Setting supprof to values' + str_hy.replace('_', ' '))
#
#            df = self.ml.m.df_profsupply_soy
#            df = df[['sy', 'pp_id', 'ca_id'] + ['value' + str_hy]]
#            df = df.set_index(['sy', 'pp_id', 'ca_id'])
#            dct_supprof = df['value' + str_hy].to_dict()
#
#            print(msg)
#            for kk, vv in dct_supprof.items():
#                self.ml.m.supprof[kk] = vv
        #######################################################################
        
         # resetting everything to base year values
        # Note: inflow profiles are static and scaled by erg_inp in the constraint


#        mask_sfh = self.ml.m.df_def_plant.pp.str.contains('SFH')
#        excl_pp = self.ml.m.df_def_plant.loc[mask_sfh].pp_id.tolist()
#        excl_pp = list_sto_pho_arch  # for pv and st opt
        # excl_pp = list_all_add  # for all add
        excl_pp = list_all_add + list_dhw_bo + list_dhw_sto  # for all add and DHW BO and STO

#        excl_pp = list_sto_arch # for stopt
#        excl_pp = list_pho_arch # exclude PHO pp for archetypes not for CH0  # for pvopt
#        excl_fl = []

        set_fuel_prices(str_fy)
        # set_curtailment_prices(str_fy)
        set_cap_pwr_leg(slct_ch, str_fy, excl_pp=excl_pp)
        set_cap_avlb(str_fy)
        set_dmnd(slct_hp, slct_rf, slct_dpf, slct_h2tr, str_fy) #slct_ev,
        set_co2_price(str_fy)
        set_erg_inp(slct_ch, str_fy)
        set_erg_chp(str_fy)
        set_fc_cap(str_fy)
        set_fc_om(str_fy)
        set_cap_trm_leg(slct_tr, str_fy)
#        set_supprof(str_hy)
#        set_priceprof(str_hy)
        set_dhw_capacity(slct_hp, str_fy)
        set_hp_capacity(slct_hp, slct_rf, str_fy)
        

        self.ml.dct_vl['swfy_vl'] = 'yr' + str(dict_fy[slct_fy])
        
    def keep_cap_new(self, dict_fy=None): 
        ''' Keep the output capacity for power plants in set add for selected year '''
        
#        df_pho = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('_PHO')]
#        list_pho_arch = df_pho.loc[~df_pho.pp.str.contains('SOL')].pp_id.tolist()
#        
#        df_sto = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.pp.str.contains('_STO')]
#        list_sto_arch = df_sto.loc[~df_sto.pp.str.contains('HYD')].pp_id.tolist()
#        
#        list_sto_pho_arch = list_pho_arch + list_sto_arch
        list_all_add = self.ml.m.df_def_plant.loc[self.ml.m.df_def_plant.set_def_add == 1].pp_id.tolist()

        
        if dict_fy is None:
            dict_fy = {
                       0:  2015,
                       1:  2020,
                       2:  2025,
                       3:  2030,
                       4:  2035,
                       5:  2040,
                       6:  2045,
                       7:  2050,    
                      }
        
        slct_fy = self.ml.dct_step['swfy']
        fy_val = dict_fy[slct_fy]
#        self.ml.dct_vl['swch_vl'] = slct_ch
            
        if fy_val == 2015:
            
            slct_col = 'cap_pwr_leg'
            msg = ('Setting cap_pwr_leg for PP ADD to values' + str(fy_val) +
                   ' from column {}'.format(slct_col))
            logger.info(msg)

            dct_cap = (self.ml.m.df_plant_encar.loc[self.ml.m.df_plant_encar.pp_id.isin(list_all_add)] # list_sto_arch for stopt
                             .set_index(['pp_id', 'ca_id'])[slct_col]
                             .to_dict())
            
            for kk, vv in dct_cap.items():
                self.ml.m.cap_pwr_leg[kk] = vv
            
        
            for kk, vv in dct_cap.items():
                self.ml.m.cap_pwr_new[kk] = 0
                
            self.ml.m.cap_pwr_new.fix()
            
        else:
            
            self.ml.m.cap_pwr_new.unfix()
            
            slct_col = 'var_yr_cap_pwr_new'
            msg = ('Setting previous capacity added for PP ADD ' + str(fy_val) +
                   ' from column {}'.format(slct_col))
            logger.info(msg)
        
    
            filt = [(key, [val]) for key, val in self.ml.dct_step.items()
                 if not key == 'swfy'] + [('swfy', [self.ml.dct_step['swfy'] - 1])]
               
            if self.ml.io.modwr.output_target == 'hdf5':
                qry = ' & '.join(str(f[0] + '=' + '"%s"'%f[1][0]) for f in filt)
                slct_run_id = pd.read_hdf(self.ml.io.cl_out, 'def_run',
                                          where=qry,
                                          columns=['run_id']).run_id.tolist()
                df_cpn = pd.read_hdf(self.ml.io.cl_out, 'var_yr_cap_pwr_new',
                                    where='run_id in %s and pp_id in %s'%(str(slct_run_id),str(list_all_add))) #list_sto_arch for stopt
        
                df_cpl = pd.read_hdf(self.ml.io.cl_out, 'par_cap_pwr_leg',
                                    where='run_id in %s and pp_id in %s'%(str(slct_run_id),str(list_all_add)))
                
                dict_cpn_add = (df_cpn.set_index(['pp_id', 'ca_id'])['value'] 
                                + df_cpl.set_index(['pp_id', 'ca_id'])['value']).to_dict()
                
            elif self.ml.io.modwr.output_target == 'psql':
                raise RuntimeError('exogenous_node_ops: Fix nd_weight for psql')
 
            elif self.ml.io.modwr.output_target == 'fastparquet':
                qry = ' & '.join(str(f[0] + '==' + '%s'%f[1][0]) for f in filt)
                
                # if mp.active_children():
                if not mp.current_process().name == 'MainProcess':
                    msg = ('MULTIPROCESSING')
                    logger.info(msg)
                    df_def_run = pd.concat([pd.read_csv(fn) for fn in glob(self.ml.io.cl_out + '/def_run*.csv')])
                
                else:
                    msg = ('NO MULTIPROCESSING')
                    logger.info(msg)
                    df_def_run= pd.read_parquet(self.ml.io.cl_out + '/def_run.parq')
                                
                slct_run_id = df_def_run.query(qry)#.to_list()
                assert len(slct_run_id) == 1, 'Found multiple run_ids'
                slct_run_id = slct_run_id.run_id.reset_index(drop=True).get(0)
                
                df_cpn = pd.read_parquet(self.ml.io.cl_out + '/var_yr_cap_pwr_new' + '_{:04d}.parq'.format(slct_run_id))
                df_cpn = df_cpn.loc[df_cpn.pp_id.isin(list_all_add)]
                                
                df_cpl = pd.read_parquet(self.ml.io.cl_out + '/par_cap_pwr_leg' + '_{:04d}.parq'.format(slct_run_id))
                df_cpl = df_cpl.loc[df_cpl.pp_id.isin(list_all_add)]

                
                dict_cpn_add = (df_cpn.set_index(['pp_id', 'ca_id'])['value'] 
                                + df_cpl.set_index(['pp_id', 'ca_id'])['value']).to_dict()
                
            for kk, vv in dict_cpn_add.items():
                self.ml.m.cap_pwr_leg[kk] = vv 