# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 11:30:33 2019

@author: rojay
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import regularizers
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing as pr
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

#Import training data
tr_well=pd.read_csv('train.csv')
#Only extract the relevant feautures (columns) of the data
tr_well_fs=tr_well[['WellType',
       'SurfAbandonDate',
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'RigReleaseDate', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD', 'WellProfile', 'WellSymbPt1', 'ProjectedDepth','StatusDate',
       '_Max`Prod`(BOE)']]

#Import test data
te_well=pd.read_csv('test.csv')
te_well_fs=te_well[['WellType', 
       'SurfAbandonDate',
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'RigReleaseDate', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD', 'WellProfile', 'WellSymbPt1', 'ProjectedDepth','StatusDate',
       '_Max`Prod`(BOE)']]

#Import validation data (which we will use as more training data)
val_well=pd.read_csv('validation.csv')
#Sort the well data by asset id so it matches the classification column
val_well=val_well.sort_values(by='EPAssetsId')
val_well_fs=val_well[['WellType', 
       'SurfAbandonDate',
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'RigReleaseDate', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD', 'WellProfile', 'WellSymbPt1', 'ProjectedDepth','StatusDate',
       '_Max`Prod`(BOE)']]




tr_well_fs= tr_well_fs.append(val_well_fs,ignore_index=True)

# Import the classfication status for each well for training and validation
well_class=pd.read_csv('Well_class_train.csv')

well_class_val=pd.read_csv('Well_class_validate.csv')

well_class= well_class.append(well_class_val)

#full_tr= pd.merge(tr_well_fs,well_class,on=['EPAssetsId'])
full_tr= tr_well_fs.iloc[:,:]
full_te= te_well_fs.iloc[:,:]
#full_val = val_well_fs.iloc[:,:]

#Convert dates to datetime objects ( so we can extract year, month ect. )
full_tr[['StatusDate']]= pd.to_datetime(full_tr['StatusDate'])
full_tr[['RigReleaseDate']]= pd.to_datetime(full_tr['RigReleaseDate'])


full_te[['StatusDate']]= pd.to_datetime(full_te['StatusDate'])
full_te[['RigReleaseDate']]= pd.to_datetime(full_te['RigReleaseDate'])
 
#full_val[['StatusDate']]= pd.to_datetime(full_val['StatusDate'])
#full_val[['RigReleaseDate']]= pd.to_datetime(full_val['RigReleaseDate'])
 
# Create a new feature that measures the time difference between when the well was finished and its last status report
timediff_tr= full_tr['RigReleaseDate']-full_tr['StatusDate']
timediff_te=  full_te['RigReleaseDate']-full_te['StatusDate']
#timediff_val = full_val['RigReleaseDate']-full_val['StatusDate']

# Convert thethe time difference between when the well was finished and its last status report into days
full_tr['timediff']=timediff_tr.dt.days
full_te['timediff']=timediff_te.dt.days

# Create a feature that is a binary variable checking if the surface abandoment date is reported
full_tr['SurfAbandonDate'].loc[full_tr['SurfAbandonDate'].notnull()]='Yes'
full_tr['SurfAbandonDate']=full_tr['SurfAbandonDate'].fillna(value='No')

full_te['SurfAbandonDate'].loc[full_te['SurfAbandonDate'].notnull()]='Yes'
full_te['SurfAbandonDate']=full_te['SurfAbandonDate'].fillna(value='No')

#full_val['SurfAbandonDate'].loc[full_val['SurfAbandonDate'].notnull()]='Yes'
#full_val['SurfAbandonDate']=full_val['SurfAbandonDate'].fillna(value='No')



dti_sd=pd.DatetimeIndex(full_tr['StatusDate'])
dti_rrd = pd.DatetimeIndex(full_tr['RigReleaseDate'])

dti_sd_te=pd.DatetimeIndex(full_te['StatusDate'])
dti_rrd_te = pd.DatetimeIndex(full_te['RigReleaseDate'])

#dti_sd_val=pd.DatetimeIndex(full_val['StatusDate'])
#dti_rrd_val = pd.DatetimeIndex(full_val['RigReleaseDate'])


# Extract month and year from last status date and well completion date
full_tr['sd_month']=dti_sd.month
full_tr['sd_year']=dti_sd.year

full_tr['rrd_month']=dti_rrd.month
full_tr['rrd_year']=dti_rrd.year

full_te['sd_month']=dti_sd_te.month
full_te['sd_year']=dti_sd_te.year

full_te['rrd_month']=dti_rrd_te.month
full_te['rrd_year']=dti_rrd_te.year


#full_val['sd_month']=dti_sd_val.month
#full_val['sd_year']=dti_sd_val.year

#full_val['rrd_month']=dti_rrd_val.month
#full_val['rrd_year']=dti_rrd_val.year


# Removes last status date and completion date from data
full_trf=full_tr.drop(['StatusDate','RigReleaseDate'],axis=1)


full_tef=full_te.drop(['StatusDate','RigReleaseDate'],axis=1)


#full_valf=full_val.drop(['StatusDate','RigReleaseDate'],axis=1)


# Make sure the factor variable well type has the same variables for both the test and training set
full_tef.loc[~full_tef['WellType'].isin(full_trf['WellType']),'WellType']='Not Applicable'

# Turn all na values into 0s
full_trf.fillna(value=0, inplace=True) 
full_tef.fillna(value=0, inplace=True) 


# Extract numerical columns and convert them all to float64
full_red=full_trf
full_red_num=full_red.loc[:,(full_red.dtypes==np.float64) | (full_red.dtypes==np.int64)]
full_red_num = full_red_num.astype('float64')


full_red_num_te=full_tef.loc[:,(full_tef.dtypes==np.float64) | (full_tef.dtypes==np.int64)]
full_red_num_te = full_red_num_te.astype('float64')

# Use a power transform to normalize the data
pt= pr.PowerTransformer()
pt.fit(full_red_num)

full_red_tfm= pt.transform(full_red_num)
full_red_tfm_te= pt.transform(full_red_num_te)

scaler = MinMaxScaler()
scaler.fit(full_red_tfm)

# Scales all data to between 0 and 1 to help optimization in machine learning 
full_red_tfm_sc= scaler.transform(full_red_tfm)
full_red_tfm_tesc= scaler.transform(full_red_tfm_te)

# Create data frame with all desired and preprocessed features
df=pd.DataFrame(full_red_tfm_sc)

tr_set=pd.concat([df, full_red[['WellType','SurfAbandonDate', 'WellProfile', 'WellSymbPt1']].reset_index(drop=True)], axis=1)
tr_set.columns= [
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD',  'ProjectedDepth',
       '_Max`Prod`(BOE)','timediff','sd_month','sd_year','rrd_month', 'rrd_year','WellType',
       'SurfAbandonDate','WellProfile', 'WellSymbPt1']

df1=pd.DataFrame(full_red_tfm_tesc)

te_set=pd.concat([df1, full_tef[['WellType','SurfAbandonDate', 'WellProfile', 'WellSymbPt1']].reset_index(drop=True)], axis=1)
te_set.columns= [
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD',  'ProjectedDepth',
       '_Max`Prod`(BOE)','timediff','sd_month','sd_year','rrd_month', 'rrd_year','WellType',
       'SurfAbandonDate','WellProfile', 'WellSymbPt1']



to_set=tr_set.append(te_set)

#One hot encode all factor variables 
to_set=pd.get_dummies(to_set)

#Create training and test set
to_set_tr = to_set.iloc[:720804,:]
to_set_te= to_set.iloc[720804:,:]

to_set_tr = to_set.iloc[:720804,:].values
to_set_te= to_set.iloc[720804:,:].values

to_set_tedf= pd.concat([te_well[['EPAssetsId']],to_set.iloc[720804:,:]],axis=1)

to_set_tedf=to_set_tedf.sort_values(by='EPAssetsId')

to_set_tedf= to_set_tedf.iloc[:,1:]

to_set_te= to_set_tedf.values

# One hot encode calssification labels (only needed for tensor flow)
y_tr=well_class[['well_status_code']].iloc[:720804,:].values.ravel()
y_train=pr.LabelBinarizer()
y_train.fit(y_tr)
y_train=y_train.transform(y_tr)

#  Set up Neural net 
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
#use softmax function for classification
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',metrics=['acc'])
model.fit(to_set_tr, y_tr,
          batch_size=24,
          epochs=30,
          validation_split = 0.05)
 
# Set up xgboost,  will do  hyperparameter tuning of max depth and learning rate
test_params = {
 'max_depth':[4,6,8], 'eta':[0.1,0.3]
}
xgb_model = xgb.sklearn.XGBClassifier()
# Do grid search for hyperparameter optimization
model = GridSearchCV(estimator = xgb_model,param_grid = test_params,scoring='f1_macro',cv=2)
#Fit model with optimized training parameters to data
model.fit(to_set_tr,y_tr)