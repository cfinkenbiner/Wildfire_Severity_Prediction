import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes

## Calculate & Plot Number of Fires per Year
def fig_fire_yr(df, start_yr, end_yr):

    years = np.arange(start_yr, end_yr) # date range

    num_fires = []
    for yr in years:
        df_yr = df[df['FIRE_YEAR'] == yr]
        num_fires.append(df_yr['OBJECTID'].nunique())
    num_fires = np.array(num_fires)

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, num_fires)

    plt.figure()
    plt.plot(years, num_fires, 'r^-')
    plt.plot(years, years*slope+intercept, 'k-')

    plt.title('Total Reported Fires in the U.S. by Year')
    plt.ylabel('Number of Reported Fires')

    print(r'Total Reported Fires:')
    print(r' r2 = %.3f' % r_value**2)
    print(r' p-value = %.3f' % p_value)

if __name__ == '__main__':
    fig_fire_yr(df, start_yr, end_yr)

    
## Calculate & Plot Number of Fires for Each State
def fig_fire_by_state(df, start_yr, end_yr):
    states = df['STATE'].unique()
    num_fires = []
    for s in states:
        df_s = df[df['STATE'] == s]
        num_fires.append(df_s['OBJECTID'].nunique())
    num_fires = np.array(num_fires)

    plt.figure(figsize=(12,3))
    plt.bar(states, num_fires, color='orange')

    plt.xticks(rotation = 90)
    plt.title('Total Reported Fires by State (' +str(start_yr)+' - ' +str(end_yr)+ ')')
    plt.ylabel('Number of Reported Fires')
    
if __name__ == '__main__':
    fig_fire_by_state(df, start_yr, end_yr)
    

## Calculate Acres within the Final Perimeter of the Fire
def fig_acres_final(df, start_yr, end_yr):
    years = np.arange(start_yr, end_yr) # date range

    fire_mean = []
    fire_max = []
    for yr in years:
        df_yr = df[df['FIRE_YEAR'] == yr]
        fire_mean.append(df_yr['FIRE_SIZE'].mean())
        fire_max.append(df_yr['FIRE_SIZE'].max())
    fire_mean = np.array(fire_mean)
    fire_max = np.array(fire_max)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, fire_mean)
    plt.plot(years, fire_mean, 'D-', color='purple')
    plt.plot(years, years*slope+intercept, 'k-')

    print("Average Number of Acres:")
    print(r' r2 = %.3f' % r_value**2)
    print(r' p-value = %.3f' % p_value)

    plt.title('Acres within the Final Perimeter of the Fire')
    plt.ylabel('Average Number of Acres')

    plt.subplot(1,2,2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, fire_max)
    plt.plot(years, fire_max, 'gD-')
    plt.plot(years, years*slope+intercept, 'k-')

    print("Maximum Number of Acres:")
    print(r' r2 = %.3f' % r_value**2)
    print(r' p-value = %.3f' % p_value)

    plt.title('Acres within the Final Perimeter of the Fire')
    plt.ylabel('Maximum Number of Acres')
    plt.tight_layout()
    
if __name__ == '__main__':
    fig_acres_final(df, start_yr, end_yr)
    

## Calculate Number of Fires in Each Class Size for a State
def fig_fire_class(state):
    # Specify date range between 1992-2015
    start_yr = 1992
    end_yr = 2015
    years = np.arange(start_yr, end_yr) # date range
    
    df_state = pd.read_csv('data/WildFires_'+state+'.csv', low_memory=False)

    df_state['FIRE_SIZE_CLASS'].value_counts().plot(kind='bar')
    plt.title('Number of Fires in Each Size Class for '+state)
    plt.xlabel('Fire Size Class')
    plt.ylabel('Number of Fires')
    
if __name__ == '__main__':
    fig_fire_class(df, start_yr, end_yr, state)

    
def fig_soil_veg_map(state):
    # load appropriate datafile
    df_state = pd.read_csv('data/WildFires_'+state+'.csv', index_col=False, low_memory=False)
    
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

        cmap = plt.cm.tab20b  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]

        # define the bins and normalize
        bounds = np.unique(values)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

        im = ax.scatter(lon, lat, 
                        alpha=0.85, 
                        s=8,
                        c = values,
                        cmap = cmap,
                        norm = norm,
                        transform = ccrs.PlateCarree())
        axgr.cbar_axes[ii].colorbar(im)
        axgr[ii].set_title(label)

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection = projection))   

    fig = plt.figure(1,figsize = (12,10))
    axgr = AxesGrid(fig, 111, axes_class = axes_class, 
                        nrows_ncols=(1,2), 
                        axes_pad=0.90,
                        cbar_location = 'right',
                        cbar_mode = 'each',
                        cbar_pad = 0.1,
                        cbar_size = '3%',
                        label_mode = '')

    create_map(df_state['LATITUDE'].values, df_state['LONGITUDE'].values,
               df_state['SOIL_TYPE'].values,
               'Soil Texture Classification', 
               axgr[0], 0)

    create_map(df_state['LATITUDE'].values, df_state['LONGITUDE'].values,
               df_state['VEG_TYPE'].values,
               'Vegetation Classification', 
               axgr[1], 1)   
    
    plt.savefig('figures/soil_veg_map_'+state+'.png', bbox_inches = 'tight', pad_inches = 0.1)

if __name__ == '__main__':
    fig_soil_veg_map(state)
