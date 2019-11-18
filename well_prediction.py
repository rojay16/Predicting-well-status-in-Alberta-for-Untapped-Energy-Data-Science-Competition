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

tr_well=pd.read_csv('train.csv')
tr_well_fs=tr_well[['WellType',
       'SurfAbandonDate',
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'RigReleaseDate', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD', 'WellProfile', 'WellSymbPt1', 'ProjectedDepth','StatusDate',
       '_Max`Prod`(BOE)']]

te_well=pd.read_csv('test.csv')
te_well_fs=te_well[['WellType', 
       'SurfAbandonDate',
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'RigReleaseDate', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD', 'WellProfile', 'WellSymbPt1', 'ProjectedDepth','StatusDate',
       '_Max`Prod`(BOE)']]


val_well=pd.read_csv('validation.csv')
val_well=val_well.sort_values(by='EPAssetsId')
val_well_fs=val_well[['WellType', 
       'SurfAbandonDate',
       'BH_Longitude', 'BH_Latitude', 'GroundElevation', 'KBElevation',
       'TotalDepth', 'RigReleaseDate', 'DaysDrilling', 'DrillMetresPerDay',
       'TVD', 'WellProfile', 'WellSymbPt1', 'ProjectedDepth','StatusDate',
       '_Max`Prod`(BOE)']]




tr_well_fs= tr_well_fs.append(val_well_fs,ignore_index=True)


well_class=pd.read_csv('Well_class_train.csv')

well_class_val=pd.read_csv('Well_class_validate.csv')

well_class= well_class.append(well_class_val)

#full_tr= pd.merge(tr_well_fs,well_class,on=['EPAssetsId'])
full_tr= tr_well_fs.iloc[:,:]
full_te= te_well_fs.iloc[:,:]
#full_val = val_well_fs.iloc[:,:]

full_tr[['StatusDate']]= pd.to_datetime(full_tr['StatusDate'])
full_tr[['RigReleaseDate']]= pd.to_datetime(full_tr['RigReleaseDate'])


full_te[['StatusDate']]= pd.to_datetime(full_te['StatusDate'])
full_te[['RigReleaseDate']]= pd.to_datetime(full_te['RigReleaseDate'])
 
#full_val[['StatusDate']]= pd.to_datetime(full_val['StatusDate'])
#full_val[['RigReleaseDate']]= pd.to_datetime(full_val['RigReleaseDate'])
 
timediff_tr= full_tr['RigReleaseDate']-full_tr['StatusDate']
timediff_te=  full_te['RigReleaseDate']-full_te['StatusDate']
#timediff_val = full_val['RigReleaseDate']-full_val['StatusDate']

full_tr['timediff']=timediff_tr.dt.days
full_te['timediff']=timediff_te.dt.days


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


full_trf=full_tr.drop(['StatusDate','RigReleaseDate'],axis=1)


full_tef=full_te.drop(['StatusDate','RigReleaseDate'],axis=1)


#full_valf=full_val.drop(['StatusDate','RigReleaseDate'],axis=1)



full_tef.loc[~full_tef['WellType'].isin(full_trf['WellType']),'WellType']='Not Applicable'

full_trf.fillna(value=0, inplace=True) 
full_tef.fillna(value=0, inplace=True) 



full_red=full_trf
full_red_num=full_red.loc[:,(full_red.dtypes==np.float64) | (full_red.dtypes==np.int64)]
full_red_num = full_red_num.astype('float64')


full_red_num_te=full_tef.loc[:,(full_tef.dtypes==np.float64) | (full_tef.dtypes==np.int64)]
full_red_num_te = full_red_num_te.astype('float64')

pt= pr.PowerTransformer()
pt.fit(full_red_num)

full_red_tfm= pt.transform(full_red_num)
full_red_tfm_te= pt.transform(full_red_num_te)

scaler = MinMaxScaler()
scaler.fit(full_red_tfm)

full_red_tfm_sc= scaler.transform(full_red_tfm)
full_red_tfm_tesc= scaler.transform(full_red_tfm_te)


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

to_set=pd.get_dummies(to_set)

to_set_tr = to_set.iloc[:720804,:]
to_set_te= to_set.iloc[720804:,:]

to_set_tr = to_set.iloc[:720804,:].values
to_set_te= to_set.iloc[720804:,:].values

to_set_tedf= pd.concat([te_well[['EPAssetsId']],to_set.iloc[720804:,:]],axis=1)

to_set_tedf=to_set_tedf.sort_values(by='EPAssetsId')

to_set_tedf= to_set_tedf.iloc[:,1:]

to_set_te= to_set_tedf.values

y_tr=well_class[['well_status_code']].iloc[:720804,:].values.ravel()
y_train=pr.LabelBinarizer()
y_train.fit(y_tr)
y_train=y_train.transform(y_tr)

model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',metrics=['acc'])
model.fit(to_set_tr, y_tr,
          batch_size=24,
          epochs=30,
          validation_split = 0.05)
 
test_params = {
 'max_depth':[4,6,8], 'eta':[0.1,0.3]
}
xgb_model = xgb.sklearn.XGBClassifier()
model = GridSearchCV(estimator = xgb_model,param_grid = test_params,scoring='f1_macro',cv=2)
model.fit(to_set_tr,y_tr)