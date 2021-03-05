# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn.metrics as metrics 
from sklearn.model_selection import KFold
from sklearn import preprocessing 

variables = ['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource',
       'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter', 'UsageBand',
       'saledate', 'fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc',
       'fiModelSeries', 'fiModelDescriptor', 'ProductSize',
       'fiProductClassDesc', 'state', 'ProductGroup', 'ProductGroupDesc',
       'Drive_System', 'Enclosure', 'Forks', 'Pad_Type', 'Ride_Control',
       'Stick', 'Transmission', 'Turbocharged', 'Blade_Extension',
       'Blade_Width', 'Enclosure_Type', 'Engine_Horsepower', 'Hydraulics',
       'Pushblock', 'Ripper', 'Scarifier', 'Tip_Control', 'Tire_Size',
       'Coupler', 'Coupler_System', 'Grouser_Tracks', 'Hydraulics_Flow',
       'Track_Type', 'Undercarriage_Pad_Width', 'Stick_Length', 'Thumb',
       'Pattern_Changer', 'Grouser_Type', 'Backhoe_Mounting', 'Blade_Type',
       'Travel_Controls', 'Differential_Type', 'Steering_Controls']


# Categorical values encoder 
def lable_encoder(df, columns):
    Encoder = dict()

    for var in columns:
        Encoder[var] = preprocessing.LabelEncoder().fit(df[var].astype(str))
        
    return Encoder

# Encode columns and remove old 
def encode_columns(df, columns, encoder = None):
    if not encoder:
        encoder = lable_encoder(df, columns)
 
    for var in columns: 
        df[var] = encoder[var].transform(df[var].astype(str))
        
    return encoder


def label_bin(df, columns):
    Encoder = dict()

    for var in columns:
        Encoder[var] = LabelBinarizer().fit(df[var].astype(str))
        
    return Encoder

def bin_columns(df, columns, encoder = None):
    if not encoder:
        encoder = label_bin(df, columns)
    
    for var in columns: 
        df[var] = encoder[var].transform(df[var].astype(str))

    return encoder



import statsmodels.api as sm
def summary_model(X, y, label='scatter'):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    summary = model.summary()
    return summary

def plot_model(X, y, label='Residual Plot'):
    model = sm.OLS(y, X).fit()
    student_resids = model.outlier_test()['student_resid']
    y_hats = model.predict(X)

    plt.scatter(y_hats, student_resids, alpha = .35, label=label)
    plt.legend()
    plt.show()


train  = pd.read_csv('data/train_1.csv')
subset_train = train.iloc[:,36:54]
cols = subset_train.columns
encode_columns(subset_train, cols, encoder = None);subset_train



'''
MachineHoursCurrentMeter has 171147 (84.8%) missing values	Missing
UsageBand has 174073 (86.2%) missing values	Missing
fiSecondaryDesc has 53172 (26.3%) missing values	Missing
fiModelSeries has 188007 (93.2%) missing values	Missing
fiModelDescriptor has 169366 (83.9%) missing values	Missing
ProductSize has 109414 (54.2%) missing values	Missing
Drive_System has 138782 (68.8%) missing values	Missing
Forks has 101959 (50.5%) missing values	Missing
Pad_Type has 154449 (76.5%) missing values	Missing
Ride_Control has 114666 (56.8%) missing values	Missing
Stick has 154449 (76.5%) missing values	Missing
Transmission has 93349 (46.3%) missing values	Missing
Turbocharged has 154449 (76.5%) missing values	Missing
Blade_Extension has 186171 (92.2%) missing values	Missing
Blade_Width has 186171 (92.2%) missing values	Missing
Enclosure_Type has 186171 (92.2%) missing values	Missing
Engine_Horsepower has 186171 (92.2%) missing values	Missing
Hydraulics has 47984 (23.8%) missing values	Missing
Pushblock has 186171 (92.2%) missing values	Missing
Ripper has 140728 (69.7%) missing values	Missing
Scarifier has 186165 (92.2%) missing values	Missing
Tip_Control has 186171 (92.2%) missing values	Missing
Tire_Size has 146451 (72.6%) missing values	Missing
Coupler has 109400 (54.2%) missing values	Missing
Coupler_System has 189099 (93.7%) missing values	Missing
Grouser_Tracks has 189161 (93.7%) missing values	Missing
Hydraulics_Flow has 189161 (93.7%) missing values	Missing
Track_Type has 161938 (80.2%) missing values	Missing
Undercarriage_Pad_Width has 161828 (80.2%) missing values	Missing
Stick_Length has 161920 (80.2%) missing values	Missing
Thumb has 161912 (80.2%) missing values	Missing
Pattern_Changer has 161920 (80.2%) missing values	Missing
Grouser_Type has 161938 (80.2%) missing values	Missing
Backhoe_Mounting has 156438 (77.5%) missing values	Missing
Blade_Type has 156373 (77.5%) missing values	Missing
Travel_Controls has 156371 (77.5%) missing values	Missing
Differential_Type has 162119 (80.3%) missing values	Missing
Steering_Controls has 162133 (80.3%) missing values	Missing
SalesID has unique values	Unique
fiBaseModel is an unsupported type, check if it needs cleaning or further analysis	Unsupported
MachineHoursCurrentMeter has 2926 (1.4%) zeros	Zeros
'''



def drop_ID_columns(df):
    better_df = df.drop(['SalesID', 'MachineID', 'ModelID'], axis=1)

x = my_work['ProductGroupDesc']
x = x.astype(str)
le = LabelEncoder()
le.fit(x)
le.transform(x)

