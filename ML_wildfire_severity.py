import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def ML_model(state):

    df_state = pd.read_csv('data/WildFires_'+state+'.csv', index_col=False, low_memory=False)
    
    ## Missing Values
    #df_state.isna().sum()
    
    ## Split into training & testing datasets
    train_data, test_data = train_test_split(df_state, test_size=0.2, random_state=25)

    print(f"Number of training examples: {train_data.shape[0]}")
    print(f"Number of testing examples: {test_data.shape[0]}")
    
    y_train = train_data['FIRE_SIZE_CLASS']
    y_test = test_data['FIRE_SIZE_CLASS']

    X_train = train_data.drop(['FIRE_SIZE', 'FIRE_SIZE_CLASS'], axis=1)
    x_test = test_data.drop(['FIRE_SIZE', 'FIRE_SIZE_CLASS'], axis=1)
    

    
    # Selected string columns with no missing values
    categorical_columns = ['FPA_ID', 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY', 
                           'NWCG_REPORTING_UNIT_ID', 'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT_NAME',
                           'OWNER_DESCR', 'STAT_CAUSE_DESCR', 'STATE']
    
    scale_columns = ['FOD_ID', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 'STAT_CAUSE_CODE', 'OWNER_CODE', 
                    'LATITUDE', 'LONGITUDE']

    numeric_columns = ['VEG_TYPE', 'SOIL_TYPE']

    features = ColumnTransformer([
        #('categorical', OneHotEncoder(handle_unknown = 'ignore'), categorical_columns),
        ('scaler', StandardScaler(), scale_columns),
        ('numeric', 'passthrough', numeric_columns)
        ])

    est = Pipeline([
        ('features', features),
        ('estimator', RandomForestClassifier(n_estimators = 100, 
                                             max_depth = 25, 
                                             warm_start = True))
        ])

    # Estimator
    est.fit(X_train, y_train)

    # save the model to disk
    # pickle.dump(est, open('finalized_ML_model_'+state+'.sav', 'wb'))

    # load the model from disk
    # est = pickle.load(open('finalized_ML_model_'+state+'.sav', 'rb'))
    
    # Predictions
    y_pred = est.predict(x_test)
    
    return x_test, y_test, y_pred


# For plotting predicted fire acres into classes
def fire_num_calc(acres):
    output = []
    for a in acres:
        if a == 'A':
            output.append(1)
        elif a == 'B':
            output.append(2)  
        elif a == 'C':
            output.append(3)   
        elif a == 'D':
            output.append(4)  
        elif a == 'E':
            output.append(5)  
        elif a == 'F':
            output.append(6) 
        elif a == 'G':
            output.append(7)
        else:
            output.append(1)

    return np.array(output)


# Translates prediction into fire classes
def fire_class_calc(acres):
    output = []
    for a in acres:
        if a < 0.25:
            output.append('A')
        elif 0.25 <= a < 10:
            output.append('B')  
        elif 10 <= a < 100:
            output.append('C')   
        elif 100 <= a < 300:
            output.append('D')  
        elif 300 <= a < 1000:
            output.append('E')  
        elif 1000 <= a < 5000:
            output.append('F') 
        elif a > 5000:
            output.append('G')
        else:
            output.append('A')
                
    return np.array(output)


def plot_predictions(state, x_test, y_test, y_pred):
    if state == 'CA':
        lat_lon = [-125, -113, 30, 43] 
    if state == 'OR':    
        lat_lon = [-125, -115, 41, 47] 
    if state == 'WA':    
        lat_lon = [-126, -116.5, 45, 50] 
    if state == 'NV':    
        lat_lon = [-121, -113, 34, 43] 

    def create_map(lat, lon, values, label, current_subplot, ii):
        ax = current_subplot
        ax.set_extent(lat_lon, ccrs.Geodetic())
        gl = ax.gridlines(linestyle='--', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False
        ax.coastlines()

        im = ax.scatter(lon, lat, 
                        alpha=0.85, 
                        s=8,
                        c = values,
                        cmap= plt.cm.get_cmap('plasma', 7),
                        transform = ccrs.PlateCarree())

        cbar = axgr.cbar_axes[ii].colorbar(im)
        cbar.set_ticks([1, 2, 3, 4, 5, 6, 7])
        cbar.set_ticklabels(["A", "B", "C", "D", "E", "F", "G"])
        axgr[ii].set_title(label)

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection = projection))   

    fig = plt.figure(1,figsize = (12,10))
    axgr = AxesGrid(fig, 111, axes_class = axes_class, 
                        nrows_ncols=(1,2), 
                        axes_pad=0.90,
                        share_all = True,
                        cbar_location = 'right',
                        cbar_mode = 'edge',
                        direction = 'row',
                        cbar_pad = 0.1,
                        cbar_size = '3%',
                        label_mode = '')

    create_map(x_test['LATITUDE'].values, x_test['LONGITUDE'].values,
               fire_num_calc(y_test),
               'Historical Recorded Fire Class', 
               axgr[0], 0)

    create_map(x_test['LATITUDE'].values, x_test['LONGITUDE'].values,
               fire_num_calc(y_pred),
               'Predicted Fire Class', 
               axgr[1], 1)   
    
    plt.tight_layout()
    plt.savefig('figures/ML_prediction_test_'+state+'.png', bbox_inches = 'tight', pad_inches = 0.1)
    
    
def calc_model_stats(y_test, y_pred):
    print('\n Accuracy: ', accuracy_score(y_test, y_pred))
    
    
def make_predictions(state):
    
    # test dataset and prediction on test
    x_test, y_test, y_pred = ML_model(state)
    
    # Make figure
    plot_predictions(state, x_test, y_test, y_pred)
    
    # Calculate Statistics
    calc_model_stats(y_test, y_pred)
    
if __name__ == '__main__':
    make_predictions(state)
    