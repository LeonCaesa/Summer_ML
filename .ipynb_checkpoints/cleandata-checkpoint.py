#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Sun Jun  7 14:57:04 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
sns.set(color_codes=True)


class cleandata:

    def __init__(self, raw_data, method='mean', selected_columns=None, small_sample=False):
        """
            Method to initliaze data cleanning

            param: raw_data, raw data outputted from sas as pandas dataframe, must have columns Date in sas format

            param: method, method used to fill in missing value

            param: seletec_clumns, a set of columns specified by the user, default is using all the data
        """
        if small_sample:
            raw_data = raw_data.iloc[-1000:,:]
        
        self.fill_in_method = method


        if selected_columns is not None:
            try:
                self.raw_data = raw_data[selected_columns]

            except:

                print(
                    'Warnings: raw_data does not have all the specified selected columns')

                self.raw_data = raw_data
        else:
            self.raw_data = raw_data
        
    def get_float_factor(self):
        
        """
            Method to get continues predictive factors
        """
        
        self.time_format()
        non_float = self.raw_data.dtypes[self.raw_data.dtypes !=
                                 'float64'].index                         
        data_float = self.raw_data.drop(non_float, axis=1)        
        self.data_float = data_float.dropna(how='any', axis=1)
        return self.data_float        
        
    def time_format(self):
        """
            Method to modify sas formatted data into python
        """
        self.raw_data['Date'] = pd.to_timedelta(
            self.raw_data['DATE'], unit='D') + pd.datetime(1960, 1, 1)

    def fill_in_missing(self, method='mean'):
        """
            Method to fill in missing data by using method
        """
        if self.fill_in_method == 'mean':

            self.raw_data.fillna(self.raw_data.mean(), inplace=True)
            
        elif self.fill_in_method == 'median':

            self.raw_data.fillna(self.raw_data.median(), inplace=True)
            
            
    def calc_vif(self, show_dist=False):
        """
            Method to calculate varaince inflation factors
        """
        self.time_format()

        data_float = self.get_float_factor()
        
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(
            data_float.values, i) for i in range(data_float.shape[1])]
        vif["features"] = data_float.columns
        
        return vif
    
    def del_missing_indicators(self):
        """
            Method to drop indicator variables that have missing data (implemented since fill_in_method doesn't apply for indicator variables)
        """

        indx_null_indicator = np.isnan(self.raw_data["rd"])
        self.raw_data = self.raw_data.loc[~indx_null_indicator, :].reset_index(
            drop=True)

    def __main__(self):

        self.time_format()
        print('time formart correction completed')

        self.del_missing_indicators()
        print('missing indicator variables deleted')

        self.fill_in_missing()
        print('missing value filled in with ' + str(self.fill_in_method) + '')

        return self.raw_data

if __name__ == '__main__':
    col_selected = pd.read_excel("selected_column.xlsx")
    col_set = set(col_selected['col_name'])
    
    
    
    raw_data = pd.read_sas('rpsdata_rfs.sas7bdat')
    df = cleandata(raw_data, selected_columns=col_set, small_sample=True)
    clean_data = df.__main__()

    # To show vif distribution
    vif_df = df.calc_vif()
    vif_df.plot.density()





