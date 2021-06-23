#--POI Activity Consolidation Algorithm--

from Utilities import pandas_to_spark
import pandas as pd
import os
import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession.0
# only run after findspark.init()
path = r'C:\Dev\nilufer\ActivityPOI-Data'
fileName_unlabel = 'Unlabelledactivities1.csv'
fileName_label = 'LabelledData.csv'
fileName_poi = 'POIs (1).xlsx'

os.chdir(path)
xl = pd.ExcelFile(fileName_poi)
# print (xl.sheet_names)
POI_df = xl.parse("Sheet1", converters={'W_Closing1': str, 'W_Opening1': str})
##POI wrangling
##'W_Closing1' +values to be converted into 2400
POI_df.loc[POI_df['W_Closing1'].str.contains('\+'), 'W_Closing1'] = '2400'
POI_df['W_Closing1'] = POI_df['W_Closing1'].astype(int)
POI_df['W_Opening1'] = POI_df['W_Opening1'].astype(int)

labelled_df = pd.read_csv(fileName_label, delimiter=',')
labelled_df.insert(0, 'activity_id', range(1000, 1000 + len(labelled_df)))

labelled_df["StartTime"] = labelled_df["StartTime"] * 100
labelled_df["EndTime"] = labelled_df["EndTime"] * 100

labelled_df['StartTime'] = labelled_df['StartTime'].astype(int)
labelled_df['EndTime'] = labelled_df['EndTime'].astype(int)

# create spark session
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

print('transform pandas DF to spark DF')
labelled_df = labelled_df.rename(columns={'ActivityLocation': 'Station'})

print(labelled_df.shape)
print(labelled_df.columns)
print(POI_df.shape)
print(POI_df.columns)

# Pandas to Spark
poi_sdf = pandas_to_spark(spark, POI_df)
labelled_sdf = pandas_to_spark(spark, labelled_df)

# labelled_df = labelled_df.head(150000)

merged_sdf = labelled_sdf.join(poi_sdf, "Station", "inner")

# merged_sdf = labelled_sdf.join(poi_sdf, labelled_sdf("Station") === poi_sdf("Station"), "inner")

print('Output', merged_sdf.count(), len(merged_sdf.columns))
print(merged_sdf.head(2))

# remove activities that span two days
# merged_sdf = merged_df[merged_df["Activity Start Date/Time"].str[0:2] ==
#                      merged_df["Activity End Date/Time"].str[0:2]]

# filter POIs based on the time of the activity
merged_sdf = merged_sdf[(merged_sdf['W_Opening1'] < merged_sdf['StartTime']) &
                        (merged_sdf['W_Closing1'] > merged_sdf['EndTime'])]

print('Output1', merged_sdf.count(), len(merged_sdf.columns))

merged_sdf = merged_sdf[
    ['UserID', 'activity_id','ActivityDate','ActivityDay', 'category_0', 'checkinsCo', 'Station', 'ActivityDuration', 'ActivityDuration(hr)',
     'StartTime', 'EndTime', 'Labelled_Acts']]
merged_sdf.head(2)

# grp_sdf = merged_sdf.groupby(['UserID','activity_id','Station','ActivityDuration','ActivityDuration(hr)','StartTime','EndTime','Labelled_Acts'])['category_0'].value_counts()

# grp_sdf = merged_sdf.groupBy('UserID','activity_id','Station','ActivityDuration','ActivityDuration(hr)','StartTime','EndTime','Labelled_Acts','category_0').count()

gexprs = ['UserID', 'activity_id','ActivityDate','ActivityDay', 'Station', 'ActivityDuration', 'ActivityDuration(hr)', 'StartTime', 'EndTime',
          'Labelled_Acts']

grp_sdf_cnt = merged_sdf.groupBy(*gexprs).pivot("category_0").count()
grp_sdf_cnt_chkin = merged_sdf.groupBy(*gexprs).pivot("category_0").sum('checkinsCo')

print('Output1', grp_sdf_cnt.count(), len(grp_sdf_cnt.columns))
print('Output2', grp_sdf_cnt_chkin.count(), len(grp_sdf_cnt_chkin.columns))

print('completed')

grp_df_cnt = grp_sdf_cnt.toPandas()
grp_df_cnt_chkin = grp_sdf_cnt_chkin.toPandas()

# create excel writer object
writer = pd.ExcelWriter('output.xlsx')
# write dataframe to excel
grp_df_cnt.to_excel(writer, 'counts')
grp_df_cnt_chkin.to_excel(writer, 'check-ins')
# save the excel
writer.save()
print('DataFrame is written successfully to Excel File.')
