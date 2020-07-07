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

from cleandata import *  # analysis:ignore
import pandas as pd
from SVM import *  # analysis:ignore
from datetime import datetime
from dateutil.relativedelta import relativedelta


class MLEngineer(cleandata):  # analysis:ignore
    def __init__(self, raw_data, num_stocks, trading_window, algorithm, method='mean', selected_columns=None, small_sample=False):
        """
            Method to initalize Machine Learning Algorithm Object,
            param: raw_data, raw data outputted from sas as pandas dataframe, must have columns Date in sas format
            param: method, method used to fill in missing value
            param: num_stocks, number of stocks in the optimial portfolio            
            param: trading window, trading frequency with unit day            
            param: algorithm, string variable indicating what algorithm wants to implement

        """

        cleandata.__init__(self, raw_data, method,
                           selected_columns, small_sample)  # analysis:ignore

        self.num_stocks = num_stocks
        self.trading_window = trading_window
        if algorithm == 'SVM':
            self.model = SVM_Machine  # analysis:ignore
        self.descriptive = ['Unnamed: 0', 'permno', 'gvkey', 'fyear']

    def return_prediction(self, time_step):
        """
            Method to predict returns
        """

        self.x_train, self.y_train, self.x_test, self.y_test = self.split_date(
            time_step)

        model = self.model(self.x_train, self.y_train,
                           self.x_test, self.y_test)
        model.tunning()
        model.training()
        y_hat = model.predict()

        return y_hat, self.y_test.values

    def split_date(self, time_step):
        '''Set looking backward dataset/train dataset.'''

        month_of_looking_back = self.trading_window
        time = pd.to_datetime(time_step).to_period('M')
        time_train_start = time - month_of_looking_back
        time_test_end = time + 1

        self.get_clean_data()
        dataset = self.clean_data

        date_list = dataset.DATE.unique()

        '''Check if looking back period has gone out of range.'''
        if time_train_start not in date_list:
            print("Invalid date, please enter a valid date. ")

        '''Check if looking ahead period has gone out of range.'''
        if time_test_end not in date_list:
            print("Invalid date, please enter a valid date. ")

        train = dataset.loc[(dataset['DATE'] > time_train_start)
                            & (dataset['DATE'] <= time)]
        test = dataset.loc[dataset['DATE'] == time_test_end]

        '''Drop gvkeys, date and all other irrelevant columns.'''
        x_train = train.drop(self.descriptive + ['RET'] + ['DATE'], axis=1)
        y_train = train['RET']
        x_test = test.drop(self.descriptive + ['RET'] + ['DATE'], axis=1)
        y_test = test['RET']

        return x_train, y_train, x_test, y_test

    def opti_stocks(self):
        """
            Method to output optimally selected stocks 
        """
        y_hat, y_test = self.return_prediction()

        indx_rank = np.argsort(y_hat)

        portfoli_permno = self.x_test.iloc[indx_rank[-self.num_stocks:]]['permno']

        return portfoli_permno.values

    def profit_anaysis(self):
        """
        Function to do performance mesaurement attribution    
        """
        port_id = set(self.opti_stocks())

        self.x_test['permno'][port_id]

    def calc_sharpe_ratio(self):
        shortrate_data = pd.read_excel("macro_data.xlsx", "short_rate")
        shortrate_data = shortrate_data[::-1].reset_index(drop=True)

        start_date = self.ret_df['test_date'].iloc[0]
        end_date = self.ret_df['test_date'].iloc[-1]

        start_index = next(i for i in range(len(shortrate_data['Date'])) if (
            shortrate_data['Date'].iloc[i].month == start_date.month) & (shortrate_data['Date'].iloc[i].year == start_date.year))
        end_index = next(i for i in range(len(shortrate_data['Date'])) if (
            shortrate_data['Date'].iloc[i].month == end_date.month) & (shortrate_data['Date'].iloc[i].year == end_date.year))

        risk_free_rate = shortrate_data.iloc[start_index:end_index +
                                             1]['short_rate'].values / 100.
        risk_free_rate = np.reshape(risk_free_rate, [len(risk_free_rate), 1])

        ret_avg = np.mean(
            self.ret_df[self.ret_cols].values * 12 - risk_free_rate, axis=0)
        ret_std = np.sqrt(
            12) * np.std(self.ret_df[self.ret_cols].values.astype(float), axis=0)
        sharpe_ratio = ret_avg / ret_std

        sharpe_ratio_result = pd.DataFrame([ret_avg, ret_std, sharpe_ratio], index=["Expected Excess Return", "Standard Deviation", "Sharpe Ratio"],
                                           columns=self.ret_cols).T
        return sharpe_ratio_result

    def calc_max_drawdown(self):
        drawdown_df = (self.ret_cumu[self.ret_cols] - self.ret_cumu[self.ret_cols].cummax(
        )) / self.ret_cumu[self.ret_cols].cummax()
        max_drawdown = drawdown_df.min(axis=0)
        return max_drawdown


if __name__ == '__main__':

    raw_data = pd.read_csv("constituents_2013_fund_tech.csv")
    ML_object = MLEngineer(raw_data, num_stocks=40,
                           trading_window=30, algorithm='SVM', small_sample=False)
    ML_object.__main__()

    self = ML_object
    port_id = ML_object.opti_stocks()

    # To show vif distribution
    vif_df = ML_object.calc_vif()
    vif_df.plot.density()


insample_rmse = self.calc_rmse(train_target, train_fit)
test_rmse = self.calc_rmse(ret_realized, ret_predict)

ret_predict = lasso.predict(test_data_input)
train_fit = lasso.predict(train_data_clean)
ret_realized = self.data_test[self.target].values


indx_rank = np.argsort(ret_predict)
security_rank = self.data_test.loc[indx_rank, 'permno']

insample_rmse = self.calc_rmse(train_target, train_fit)
test_rmse = self.calc_rmse(ret_realized, ret_predict)

ret_predict = lasso.predict(test_data_input)
train_fit = lasso.predict(train_data_clean)
ret_realized = self.data_test[self.target].values
