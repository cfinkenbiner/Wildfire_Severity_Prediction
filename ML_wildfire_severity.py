import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


def ML_model(state):

    df_state = pd.read_csv('WildFires_'+state+'.csv', index_col=False, low_memory=False)
    
    ## Missing Values
    #df_state.isna().sum()
    
    ## Split into training & testing datasets
    train_data, test_data = train_test_split(df_state, test_size=0.2, random_state=25)

    print(f"Number of training examples: {train_data.shape[0]}")
    print(f"Number of testing examples: {test_data.shape[0]}")
    
    y_train = train_data['FIRE_SIZE']
    y_test = test_data['FIRE_SIZE']

    X_train = train_data.drop(['FIRE_SIZE', 'FIRE_SIZE_CLASS'], axis=1)
    x_test = test_data.drop(['FIRE_SIZE', 'FIRE_SIZE_CLASS'], axis=1)
    
    # Selected string columns with no missing values
    categorical_columns = ['FPA_ID', 'SOURCE_SYSTEM_TYPE', 'SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY', 
                           'NWCG_REPORTING_UNIT_ID', 'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT_NAME',
                           'OWNER_DESCR', 'STAT_CAUSE_DESCR', 'STATE', 'LATITUDE', 'LONGITUDE']

    scale_columns = ['FOD_ID', 'FIRE_YEAR', 'DISCOVERY_DATE', 'DISCOVERY_DOY', 'STAT_CAUSE_CODE', 'OWNER_CODE']

    numeric_columns = ['VEG_TYPE', 'SOIL_TYPE']

    features = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown = 'ignore'), categorical_columns),
        ('scaler', StandardScaler(), scale_columns),
        ('numeric', 'passthrough', numeric_columns)
        ])

    est = Pipeline([
        ('features', features),
        ('estimator', LinearRegression())
        ])
    
    # Estimator
    est.fit(X_train, y_train)
    
    return x_test, y_test, est.predict(x_test)

# For plotting predicted fire acres into classes
def fire_num_calc(acres):
    output = []
    for a in acres:
        if a < 0.25:
            output.append(1)
        elif 0.25 <= a < 10:
            output.append(2)  
        elif 10 <= a < 100:
            output.append(3)   
        elif 100 <= a < 300:
            output.append(4)  
        elif 300 <= a < 1000:
            output.append(5)  
        elif 1000 <= a < 5000:
            output.append(6) 
        elif a > 5000:
            output.append(7)
        else:
            output.append(None)
                
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
            output.append('NA')
                
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

def make_predictions(state):
    
    # test dataset and prediction on test
    x_test, y_test, y_pred = ML_model(state)
    
    # Make figure
    print('\n Fire Class Records from and Predictions on the test dataset:')
    plot_predictions(state, x_test, y_test, y_pred)
    
if __name__ == '__main__':
    make_predictions(state)