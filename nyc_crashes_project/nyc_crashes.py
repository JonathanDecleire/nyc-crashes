'''
References
https://www.dataquest.io/blog/machine-learning-preparing-data/
https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html
https://pbpython.com/currency-cleanup.html
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.use("TKAgg")
from datetime import datetime
from dateutil import parser
import re

url = 'https://raw.githubusercontent.com/becodeorg/LIE-Thomas-1.26/master/content/additional_resources/datasets/NYC%20Motor%20Vehicle%20Crashes/data_100000.csv?token=AQYISHGQZ5YUMG2VVZJUEAC73HXKY'
df = pd.read_csv(url, low_memory=False)

# Remove all columns with more than 50% missing values
half_count = len(df) / 2
df = df.dropna(thresh=half_count, axis=1) #29 --> 21 columns

#print(df.columns)

df_dtypes = pd.DataFrame(df.dtypes, columns=['dtypes'])

'''
borough                         object
zip_code                       float64
latitude                       float64
longitude                      float64
location                        object
on_street_name                  object

Several variables that give a location of accident
We will choose on_street_name as our variable to analyze the dataset
and drop all the others.
borough and zip_code represent too wide of a location
latitude, longitude, and location represent too precise of a location
'''

drop_list = ['collision_id','borough','zip_code','latitude','longitude','location']

df = df.drop(drop_list, axis=1)
df_dtypes = pd.DataFrame(df.dtypes, columns=['dtypes'])

# killed = ['number_of_persons_killed','number_of_pedestrians_killed','number_of_cyclist_killed','number_of_motorist_killed']
# df['number_of_persons_killed'].plot.hist(alpha=0.5,bins=1)
# plt.show()
#
# print(df['number_of_persons_killed'].sum())

'''
We need to decide on a target column
Our main goal is to predict where and when car crashes will lead to
deaths and/or injuries

number_of_persons_killed could be our target column
number_of_persons_injured could be our target column
or both? combine the stats in some way?
is there a way to find whether the statistic killed is included
in injured or not?

because we want to know how deaths and injuries occur and prevent them
'''
# We create a new column number_of_persons_affected
# taking into account killed and injured
# this is our target column.
df['number_of_persons_affected'] = (df['number_of_persons_killed'] + df['number_of_persons_injured'])


#print(df.loc[df['number_of_persons_killed'] == 3])
# print(df['number_of_persons_killed'].loc[[64015]])
# print(df['number_of_pedestrians_killed'].loc[[64015]])
# print(df['number_of_cyclist_killed'].loc[[64015]])
# print(df['number_of_motorist_killed'].loc[[64015]])

# print(df['number_of_persons_injured'].loc[[64015]])
# print(df['number_of_pedestrians_injured'].loc[[64015]])
# print(df['number_of_cyclist_injured'].loc[[64015]])
# print(df['number_of_motorist_injured'].loc[[64015]])

#print(df['number_of_persons_injured'].value_counts())

### We will reformat the crash_date variable to obtain the year,
### month, and day new columns.
date_df = df['crash_date']

df['crash_date'] = df['crash_date'].astype(str)
df['month']=df['crash_date'].apply(lambda x:str(x)[5:7]).astype(int)
df['day']=df['crash_date'].apply(lambda x:str(x)[8:10]).astype(int)
df['year']=df['crash_date'].apply(lambda x:str(x)[0:4]).astype(int)
drop_crash_date = ['crash_date']
df = df.drop(drop_crash_date, axis=1)

### we will reformat the crash_time variable
### ideally we would like our day to be divided in 24 bins of one hour
### that way we can see during what time periods the most accidents occur
### times are already in 24:00 format

#https://stackoverflow.com/questions/32375471/pandas-convert-strings-to-time-without-date
df['crash_time'] = pd.to_datetime(df['crash_time'], format='%H:%M').dt.hour
time_df = df['crash_time']
#print(df['crash_time'])

#https://towardsdatascience.com/how-to-group-data-by-different-time-intervals-using-python-pandas-eb7134f9b9b0
#df.resample('H', on='crash_time').number_of_persons_killed.sum()


#print(df['crash_time'].value_counts())
#df['crash_time'].plot.hist(alpha=0.5,bins=23)
#plt.show()

df_dtypes = pd.DataFrame(df.dtypes, columns=['dtypes'])
#print(df_dtypes)

# df['month'].plot.hist(alpha=0.5,bins=12)
# plt.show()
'''
Let us check contributing_factor_vehicle_1 and 2
'''
df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].replace(
['Driver Inattention/Distraction', 'Following Too Closely', 'Failure to Yield Right-of-Way', 'Passing or Lane Usage Improper',
'Unsafe Lane Changing', 'Traffic Control Disregarded', 'Passing Too Closely', 'Backing Unsafely', 'Unsafe Speed', 'Turning Improperly',
'Driver Inexperience', 'Reaction to Uninvolved Vehicle', 'Alcohol Involvement', 'Aggressive Driving/Road Rage', 'Fell Asleep',
'Passenger Distraction', 'Outside Car Distraction', 'Steering Failure', 'Lost Consciousness', 'Failure to Keep Right',
'Illnes', 'Fatigued/Drowsy', 'Drugs (illegal)', 'Cell Phone (hand-Held)', 'Physical Disability', 'Using On Board Navigation Device',
'Eating or Drinking', 'Prescription Medication', 'Cell Phone (hands-free)', 'Other Electronic Device', 'Listening/Using Headphones'], 'Driver_fault'
)

df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].replace(
['Oversized Vehicle', 'Brakes Defective', 'Tire Failure/Inadequate', 'Accelerator Defective', 'Vehicle Vandalism', 'Tinted Windows',
'Headlights Defective', 'Tow Hitch Defective', 'Windshield Inadequate'], 'Car_defect'
)

df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].replace(
['View Obstructed/Limited', 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion', 'Pavement Slippery', 'Obstruction/Debris',
'Glare', 'Traffic Control Device Improper/Non-Working', 'Driverless/Runaway Vehicle', 'Lane Marking Improper/Inadequate',
'Animals Action', 'Pavement Defective', 'Other Lighting Defects', 'Shoulders Defective/Improper'], 'Environment_issue'
)

df['contributing_factor_vehicle_1'] = df['contributing_factor_vehicle_1'].replace(
['Unspecified', 'Other Vehicular'], 'Other'
)

#print(df['contributing_factor_vehicle_1'].nunique())

### Drop contributing_factor_vehicle_2 column because too much overlap with 1
drop_contributing_factor_vehicle_2 = ['contributing_factor_vehicle_2']
df = df.drop(drop_contributing_factor_vehicle_2, axis=1)


'''
Let us check vehicle_type_code1 and vehicle_type_code2
'''
#print(df['vehicle_type_code1'].value_counts())
preview = df['vehicle_type_code1'].value_counts()
# print(preview[0:10])
# print(preview[10:20])
# print(preview[20:30])
#print(df['vehicle_type_code1'].nunique())

# null_counts = df.isnull().sum()
# print("Number of null values in each column:\n{}".format(null_counts))

## Use dropna method to remove all rows containing any missing values
df = df.dropna()
null_counts = df.isnull().sum()
#print("Number of null values in each column:\n{}".format(null_counts))
#print(df.shape)

#print("Data types and their frequency\n{}".format(df.dtypes.value_counts()))

### Removing the spaces
df['on_street_name'] = df['on_street_name'].str.rstrip(' ').astype('str')

object_columns_df = df.select_dtypes(include=['object'])
#print(object_columns_df.iloc[0])

cols = ['on_street_name', 'contributing_factor_vehicle_1', 'contributing_factor_vehicle_2', 'vehicle_type_code1', 'vehicle_type_code2']
# for name in cols:
#     print(name, ':')
#     print(object_columns_df[name].value_counts(), '\n')

#print(df['vehicle_type_code1'].value_counts(),'\n')

# vehicle_type_code1 = df['vehicle_type_code1'].value_counts()
# vehicle_type_code1.to_csv('vehicle_type_code1.csv', index=True)

df['vehicle_type_code1'] = df['vehicle_type_code1'].astype(str).str.upper()

# Emergency
df['vehicle_type_code1'] = df['vehicle_type_code1'].replace(
['AMBULANCE', 'AMBUL', 'AMB', 'AMBU', 'FDNY AMBUL', 'GEN  AMBUL', 'AMBULACE', 'ALMBULANCE', 'LEASED AMB', 'AMBULENCE', 'ABULANCE', 'WHITE AMBU',
 'NYC AMBULA', 'FIRTRUCK', 'FIRE', 'FIRET', 'FDNY LADDE', 'FIRE ENGIN', 'FIRETRUCK', 'FDNY TRUCK', 'FDNY #226', 'FD LADDER', 'NYC FD',
 'FDNY FIRE', 'FDNY', 'FDNY ENGIN', 'FIRE TRUCK', 'FDNY EMS'], 'EMERGENCY'
)

# Two-wheels
df['vehicle_type_code1'] = df['vehicle_type_code1'].replace(
['E BIK', 'E-BIKE', 'BIKE', 'MINIBIKE', 'MOTORBIKE', 'DIRT BIKE', 'E BIKE','E SKATE BO',
'SKATEBOARD', 'E-SCOOTER', 'E-SCO', 'SCOOT', 'VESPA', 'SCOOTER', 'MOTORSCOOTER', 'MOTORSCOOT',
'CROSS', 'MOTOR SCOO', 'ELECTRIC S', 'HRSE', 'HORSE', 'MOPED', 'MINICYCLE', 'E-BIK',
'MOTORCYCLE', 'LAWNMOWER'], 'TWO-WHEELS'
)

# Car
df['vehicle_type_code1'] = df['vehicle_type_code1'].replace(
['STATION WAGON/SPORT UTILITY VEHICLE', 'SPORT UTILITY / STATION WAGON', 'UTILI',
'UTIL', 'UTILITY VE', 'UTILITY.', 'PICKU', 'PICK-', 'PU', 'PICK-UP TRUCK', 'PICKUP WITH MOUNTED CAMPER', 'PICK UP TR', 'PICK UP', 'F150XL PIC',
'F-250', 'PICK-UP TR', 'SUBURBAN', 'SUBN WHI', 'CONVERTIBLE', 'PASSENGER VEHICLE', '2 DR SEDAN',
'4 DR SEDAN', 'LIMO', 'LIMOU', 'CHEVROLET', 'SELF', 'PAS', 'E-350', 'PK', 'MINI', 'UTILITY',
'UT', 'SMART CAR', 'CITY', 'WHITE', 'SEDAN'], 'CAR'
)

#TRUCK
df['vehicle_type_code1'] = df['vehicle_type_code1'].replace(['TOW TRUCK', 'TOW T', 'TK', 'TOWTR', 'TOW TRUCK / WRECKER', 'TRK', 'TRUCK',
'CEMENT TRU', 'DUMP TRUCK', 'FREIGHT TR', '18 WHEELER', 'BUCKET TRU', 'FLATBED', 'FLATBED FR', 'FLAT BED', 'FLAT', 'FREIGHT', 'FREIGHT TR',
'TRC', 'FREIG', 'BOXTR', 'BEVERAGE TRUCK', 'DUMP', 'TRACTOR TRUCK DIESEL', 'TRACTOR TRUCK GASOLINE', 'TRACT', 'TRACTOR TR', 'TRACTOR',
'JOHN DEERE', 'BACKHOE', 'BOBCAT FOR', 'CAT', 'CAMPER TRA', 'BACK HOE', 'TRAC', 'BOX TRUCK',
'TOWER', 'GARBAGE OR REFUSE', 'CONCRETE MIXER', 'FLAT RACK', 'CARRY ALL', 'BULK AGRICULTURE',
'TANKER', 'ARMORED TRUCK', 'MOTORIZED HOME', 'TRAIL', 'MOBIL', 'FOOD TRUCK', 'FORK LIFT',
'FORKLIFT', 'LIVESTOCK RACK', 'WELL DRILLER', 'MACK', 'LADDER 34', 'STAKE OR RACK',
'GARBAGE TR', 'MECHANICAL', 'TRLR', 'LUNCH WAGON', 'BOX T', 'HI TA', 'LIFT BOOM',
'BACKH', 'BACK', 'TRAILER', 'TRUCK FLAT', 'BOX'], 'TRUCK'
)

# Public services
df['vehicle_type_code1'] = df['vehicle_type_code1'].replace(['BUS', 'SCHOO', 'SCHOOL BUS', 'MTA BUS', 'POSTAL BUS', 'POLIC', 'POLICE',
'TAXI', 'CHASSIS CAB', 'GOVERNMENT' ], 'PUBLIC SERVICES'
)

#VAN
df['vehicle_type_code1'] = df['vehicle_type_code1'].replace(
['VAN', 'VAN CAMPER', 'TRUCK VAN', 'WORK VAN', 'FORD VAN', 'USPS VAN', 'REFRIGERATED VAN', 'CARGO VAN', 'DELIVERY', 'DILEVERY T', 'DELIVERY T',
'COURIER', 'UPS TRUCK', 'MESSAGE SI', 'DELIVERY V', 'UPS TRUCK', 'MAIL TRUCK', 'POSTAL TRU', 'FREIG DELV', 'USPS POSTA', 'PAYLO', 'POSTA',
'FREIGHT FL', 'DELIV', 'VAN T', 'USPS', 'USPS TRUCK', 'DELV', 'USPS/GOVT'], 'VAN'
)

# Other
df['vehicle_type_code1'] = df['vehicle_type_code1'].replace(
['P/SH', 'COM', 'CONT', 'LTR', '3-DOOR', 'CONST', 'OTHER', 'OPEN BODY', 'COMME',
'APPOR', 'POWER', 'MULTI-WHEELED VEHICLE', 'UNKNO', 'STREE', 'ELECT', 'MTA B',
'BTM', 'POWER SHOV', 'UNK', 'PC', 'E REVEL SC', 'J1', '1C', 'WH FORD CO', 'CHEVY EXPR',
'UNKNOWN', 'COMMERCIAL', 'DOT EQUIPM', 'LCOMM', 'REFG'], 'OTHER'
)


df['vehicle_type_code1'] = df['vehicle_type_code1'].astype(str).str.lower()

# vehicle_dictionary = {
# 'car' : 0,
# 'truck' : 1,
# 'two_wheel' : 2,
# 'van' : 3,
# 'emergency' : 4,
# 'bus' : 5,
# 'taxi' : 6,
# 'other' : 7
# }

#### vehicle_type_code2 now

#vehicle_type_code2 = df['vehicle_type_code2'].value_counts()
#vehicle_type_code2.to_csv('vehicle_type_code2.csv', index=True)

df['vehicle_type_code2'] = df['vehicle_type_code2'].astype(str).str.upper()

# Emergency
df['vehicle_type_code2'] = df['vehicle_type_code2'].replace(
['AMBULANCE', 'AMBUL', 'AMB', 'AMBU', 'FDNY AMBUL', 'GEN  AMBUL', 'AMBULACE', 'ALMBULANCE', 'LEASED AMB', 'AMBULENCE', 'ABULANCE', 'WHITE AMBU',
 'NYC AMBULA', 'FIRTRUCK', 'FIRE', 'FIRET', 'FDNY LADDE', 'FIRE ENGIN', 'FIRETRUCK', 'FDNY TRUCK', 'FDNY #226', 'FD LADDER', 'NYC FD',
 'FDNY FIRE', 'FDNY', 'FDNY ENGIN', 'FIRE TRUCK', 'FDNY EMS', 'EMERGENCY', 'FDNY FIRET',
 'NYC FIRETR', 'FRT'], 'EMERGENCY'
)

# Two-wheels
df['vehicle_type_code2'] = df['vehicle_type_code2'].replace(
['E BIK', 'E-BIKE', 'BIKE', 'MINIBIKE', 'MOTORBIKE', 'DIRT BIKE', 'E BIKE','E SKATE BO',
'SKATEBOARD', 'E-SCOOTER', 'E-SCO', 'SCOOT', 'VESPA', 'SCOOTER', 'MOTORSCOOTER', 'MOTORSCOOT',
'CROSS', 'MOTOR SCOO', 'ELECTRIC S', 'HRSE', 'HORSE', 'MOPED', 'MINICYCLE', 'E-BIK',
'MOTORCYCLE', 'LAWNMOWER', 'MOTOR', 'DIRTB', 'E SCO', 'SKATE', 'E-SCOTER', 'E SCOOTER',
'DIRTBIKE', 'EBIKE', 'MOPD', 'PUSH SCOOT', 'RAZOR SCOO', 'GAS SCOOTE', 'ESCOOTER'], 'TWO-WHEELS'
)

# Car
df['vehicle_type_code2'] = df['vehicle_type_code2'].replace(
['STATION WAGON/SPORT UTILITY VEHICLE', 'SPORT UTILITY / STATION WAGON', 'UTILI',
'UTIL', 'UTILITY VE', 'UTILITY.', 'PICKU', 'PICK-', 'PU', 'PICK-UP TRUCK', 'PICKUP WITH MOUNTED CAMPER', 'PICK UP TR', 'PICK UP', 'F150XL PIC',
'F-250', 'PICK-UP TR', 'SUBURBAN', 'SUBN WHI', 'CONVERTIBLE', 'PASSENGER VEHICLE', '2 DR SEDAN',
'4 DR SEDAN', 'LIMO', 'LIMOU', 'CHEVROLET', 'SELF', 'PAS', 'E-350', 'PK', 'MINI', 'UTILITY',
'UT', 'SMART CAR', 'CITY', 'WHITE', 'SEDAN', 'SMALL COM VEH(4 TIRES) ', 'INTL', 'E350',
'FORD','BLACK', 'WAGON', 'UTIL WH', 'PICKUP', 'PICK', 'UTILITY TR'], 'CAR'
)

#TRUCK
df['vehicle_type_code2'] = df['vehicle_type_code2'].replace(
['TOW TRUCK', 'TOW T', 'TK', 'TOWTR', 'TOW TRUCK / WRECKER', 'TRK', 'TRUCK',
'CEMENT TRU', 'DUMP TRUCK', 'FREIGHT TR', '18 WHEELER', 'BUCKET TRU', 'FLATBED', 'FLATBED FR', 'FLAT BED', 'FLAT',
'FREIGHT', 'FREIGHT TR', 'TRC', 'FREIG', 'BOXTR', 'BEVERAGE TRUCK', 'DUMP', 'TRACTOR TRUCK DIESEL', 'TRACTOR TRUCK GASOLINE',
'TRACT', 'TRACTOR TR', 'TRACTOR', 'JOHN DEERE', 'BACKHOE', 'BOBCAT FOR', 'CAT', 'CAMPER TRA', 'BACK HOE', 'TRAC', 'BOX TRUCK',
'TOWER', 'GARBAGE OR REFUSE', 'CONCRETE MIXER', 'FLAT RACK', 'CARRY ALL', 'BULK AGRICULTURE',
'TANKER', 'ARMORED TRUCK', 'MOTORIZED HOME', 'TRAIL', 'MOBIL', 'FOOD TRUCK', 'FORK LIFT',
'FORKLIFT', 'LIVESTOCK RACK', 'WELL DRILLER', 'MACK', 'LADDER 34', 'STAKE OR RACK',
'GARBAGE TR', 'MECHANICAL', 'TRLR', 'LUNCH WAGON', 'BOX T', 'HI TA', 'LIFT BOOM',
'BACKH', 'BACK', 'TRAILER', 'TRUCK FLAT', 'BOX', 'CEMEN', 'TUCK', 'GARBA', 'GLASS RACK',
'SNOW PLOW', 'LADDER TRU', '18 WEELER', 'CARGO TRUC', 'TRL', 'U-HAUL', 'MTA TRUCK',
'FLEET', 'UHAUL TRUC', 'FREIGHTLIN', 'FRIEGHTLIN'], 'TRUCK'
)

# Public services
df['vehicle_type_code2'] = df['vehicle_type_code2'].replace(['BUS', 'SCHOO', 'SCHOOL BUS', 'MTA BUS', 'POSTAL BUS', 'POLIC', 'POLICE',
'TAXI', 'CHASSIS CAB', 'GOVERNMENT', 'YELLO', 'PEDICAB', 'US GOVT VE', 'HOPPER',
 'NYC BUS', 'SCHOOLBUS', 'NYC ACS VA'], 'PUBLIC SERVICES'
)

#VAN
df['vehicle_type_code2'] = df['vehicle_type_code2'].replace(
['VAN', 'VAN CAMPER', 'TRUCK VAN', 'WORK VAN', 'FORD VAN', 'USPS VAN', 'REFRIGERATED VAN', 'CARGO VAN', 'DELIVERY', 'DILEVERY T', 'DELIVERY T',
'COURIER', 'UPS TRUCK', 'MESSAGE SI', 'DELIVERY V', 'UPS TRUCK', 'MAIL TRUCK', 'POSTAL TRU', 'FREIG DELV', 'USPS POSTA', 'PAYLO', 'POSTA',
'FREIGHT FL', 'DELIV', 'VAN T', 'USPS', 'USPS TRUCK', 'DELV', 'USPS/GOVT', 'FOOD', 'MAIL',
'MOVIN', 'MAIL', 'US PO', 'LIVERY VEHICLE', 'MOVIN', 'VANETTE', 'VAN (', 'CAMPE',
'USPOS', 'MAILTRUCK', 'FD TRUCK', 'POST', 'MINI VAN', 'VAN/TRANSI', 'MOBILE'], 'VAN'
)

# Other
df['vehicle_type_code2'] = df['vehicle_type_code2'].replace(
['P/SH', 'COM', 'CONT', 'LTR', '3-DOOR', 'CONST', 'OTHER', 'OPEN BODY', 'COMME',
'APPOR', 'POWER', 'MULTI-WHEELED VEHICLE', 'UNKNO', 'STREE', 'ELECT', 'MTA B',
'BTM', 'POWER SHOV', 'UNK', 'PC', 'E REVEL SC', 'J1', '1C', 'WH FORD CO', 'CHEVY EXPR',
'UNKNOWN', 'COMMERCIAL', 'DOT EQUIPM', 'LCOMM', 'REFG', 'PSD', 'CRANE', 'FORKL', 'IP',
'FORK', 'PALLET', 'PASS', 'INTL', 'ENCLOSED BODY - NONREMOVABLE ENCLOSURE',
'T630 FORKL', 'LMA', 'ROAD SWEEP', 'DETAC', 'TRA/R', 'TRANS', 'ART M', 'ALUMI', 'WORK',
'JOHN', 'CAT32', 'SELF-', 'ENGIN', 'CHURC', 'ENCLO', 'STREET SWE', 'BACKHOE LO',
'INTERNATIO', 'REVEL', 'SANITATION', 'TILT TANDE', 'I1', 'LAWN MOWER', 'SWEEPER',
'TRANSPORT', 'C1', 'STREET CLE', 'BOB CAT', 'UKN', 'GOLF CART', '4DSD', 'COMMERCIAL',
'SPECIAL PU', 'HAUL FOR H', 'TREE CUTTE', 'RAM', 'GLP050VXEV', 'ACCESS A R', 'LCOM',
'SKID LOADE', 'ITAS', 'HEARSE', 'LIGHT TRAI', 'COMMERICAL'], 'OTHER'
)


df['vehicle_type_code2'] = df['vehicle_type_code2'].astype(str).str.lower()

#print(df['vehicle_type_code2'].unique())


# df.set_index(['number_of_persons_affected'])
# print(df.head())
cols = df.columns.tolist()
cols = cols[-4:] + cols[0:13]
df = df[cols]
#print(df.columns)



#df.to_csv('cleaned_df.csv')
cleaned_df = pd.read_csv('cleaned_df.csv')
print(cleaned_df.columns)
