import os
import pandas as pd #Library to handle with dataframes
import matplotlib.pyplot as plt # Library to plot graphics
import numpy as np # To handle with matrices
import seaborn as sns # to build modern graphics
from scipy.stats import kurtosis, skew # it's to explore some statistics of numerical values
from scipy import stats
import os
import dataframe_image as dfi



PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def resume(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]  
    summary['Uniques'] = df.nunique().values
    summary['Duplicated'] = df.duplicated().sum()
    
    for name in summary['Name'].value_counts().index:
        #summary.loc[summary['Name'] == name, 'RowCount'] = str(df[name].size)
        summary.loc[summary['Name'] == name, 'Duplicated'] = df[name].duplicated().sum()
        summary.loc[summary['Name'] == name, 'Min'] = str(df[name].min())
        summary.loc[summary['Name'] == name, 'Max'] = str(df[name].max())
        
        summary.loc[summary['Name'] == name, 'Null'] = str(df[name].isnull().sum())
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    return summary

def CalcOutliers(df_num): 
    
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Identified outliers: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return


def countplot(df, color="royalblue"):
    plt.figure(figsize=(8,5))
    sns.barplot(y=df.iloc[:,0], x=df.iloc[:,1], orient="h", color=color, order=df.sort_values("Counts",ascending = False)[df.columns[0]])
    plt.title("Top 5")
    

def save_all_countplot(df, TABLE_ID, tight_layout=True, fig_extension="png", resolution=300): 
    PROJECT_ROOT_DIR = "."
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", TABLE_ID)
    os.makedirs(IMAGES_PATH, exist_ok=True)
    
    table_df = resume(df)
    print("Saving table", TABLE_ID)
    dfi.export(table_df, "table_" + TABLE_ID + '.' + fig_extension)
    
    columns = df.columns
    for column in columns:
        df_column = df[column].value_counts().rename_axis(column).reset_index(name='Counts').iloc[:5]
        countplot(df_column)
        
        #dfi.export(, "column_" + column + '.' + fig_extension)
        
        path = os.path.join(IMAGES_PATH, column + "_count." + fig_extension)
        print("Saving figure", column)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)