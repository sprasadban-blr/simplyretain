#from RetentionApp.webapp.ConfigUtil import ConfigUtil
from ConfigUtil import ConfigUtil
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


class DataPreProcessing(object):

    def __init__(self, data, config_path):
        self.data = data  # pd.DataFrame(DataLoader.get_data())
        self.columns = self.data.columns
        config_util = ConfigUtil(config_path)
        self.config_params = config_util.get_params()
        self.categorical = []
        self.numerical = []
        self.categorical_mapping = {}
        print('Removing redundant/empty data')
        self.trim_data()
        print('Identifying categorical and numerical columns')
        self.identify_cat_num_cols()
        print('Filling missing values')
        self.fill_na()

    def get_categorical(self):
        return self.categorical

    def get_numerical(self):
        return self.numerical

    def get_categorical_mapping(self):
        return self.categorical_mapping

    def get_pre_processed_data(self):
        return self.data

    def trim_data(self):
        trim_thres = self.config_params["ignore_thres_col"]
        num_rows = len(self.data)
        # Dropping columns
        self.data.dropna(axis=1, thresh=round(1 - float(trim_thres)) * num_rows, inplace=True)
        # Dropping rows
        trim_thres = self.config_params["ignore_thres_row"]
        num_cols = len(self.columns)
        self.data.dropna(axis=0, thresh=round(1 - float(trim_thres)) * num_cols, inplace=True)

    def identify_cat_num_cols(self):
        for col in self.data.columns:
            self.refactor_categorical(col)

    def refactor_categorical(self, col):
        unique_vals = set(list(self.data[col]))
        col_dtypes = self.data.dtypes
        if len(unique_vals) < int(self.config_params['categorical_thres']):
            unique_vals_dict = {}
            new_val = 0
            for val in unique_vals:
                unique_vals_dict[val] = new_val
                new_val += 1
            self.categorical_mapping[col] = unique_vals_dict
            self.categorical.append(col)
            self.data[col] = self.data[col].apply(lambda x: unique_vals_dict[x])
            # Replace back NaN values
            if unique_vals_dict.get(np.NaN) is not None:
                self.data[col].replace(unique_vals_dict.get(np.NaN), np.NaN, inplace=True)
            # print(col, unique_vals_dict)
        elif col_dtypes[col] != 'O':
            self.numerical.append(col)

    def fill_na(self):
        imputer_num = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer_num = imputer_num.fit(self.data[self.numerical])
        self.data[self.numerical] = imputer_num.transform(self.data[self.numerical])
        imputer_cat = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        imputer_cat = imputer_cat.fit(self.data[self.categorical])
        self.data[self.categorical] = imputer_cat.transform(self.data[self.categorical])

