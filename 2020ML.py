#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Wed Jun  3 18:44:48 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""

from cleandata import *
import pandas as pd
#import os 
#os.chdir('2020_Summer_ML/Summer_ML/')
from SVM import *



class MLEngineer(cleandata):
    def __init__(self, raw_data, num_stocks, trading_window, algorithm, method='mean', selected_columns=None, small_sample=False):        
        """
            Method to initalize data prepartion
        """
        
        cleandata.__init__(self, raw_data, method,
                           selected_columns, small_sample)
        self.num_stocks = num_stocks
        self.trading_window = trading_window
        self.algorithm = algorithm

    def return_prediction(self):
        """
            Method to predict returns
        """
        
        

    def opti_stocks(self):
        """
            Method to output optimally selected stocks 
        """
        indx_rank = np.argsort(ret_predict)
        security_rank = self.data_test.loc[indx_rank,'permno']


    def profit_anaysis(self):
        
        """
        Function to do performance mesaurement attribution    
        """



import numpy as np


def PMA_analysis(pred_return, real_return):
    """
    Function to do performance mesaurement attribution
    
    """
    


insample_rmse = self.calc_rmse(train_target,train_fit)
test_rmse = self.calc_rmse(ret_realized,ret_predict)

ret_predict = lasso.predict(test_data_input)
train_fit = lasso.predict(train_data_clean)
ret_realized = self.data_test[self.target].values        





indx_rank = np.argsort(ret_predict)
security_rank = self.data_test.loc[indx_rank,'permno']

insample_rmse = self.calc_rmse(train_target,train_fit)
test_rmse = self.calc_rmse(ret_realized,ret_predict)

ret_predict = lasso.predict(test_data_input)
train_fit = lasso.predict(train_data_clean)
ret_realized = self.data_test[self.target].values  

    
if __name__ == '__main__':
#    col_selected = pd.read_excel("selected_column.xlsx")
#    col_set = set(col_selected['col_name'])
#    raw_data = pd.read_sas('rpsdata_rfs.sas7bdat')
    raw_data = pd.read_csv("data/constituents_2013_fund_tech.csv")
    df = MLEngineer(raw_data, 40, 20, 1, small_sample= True)
    clean_data = df.__main__()

    # To show vif distribution
#    vif_df = df.calc_vif()
#    vif_df.plot.density()
