# Databricks notebook source
# MAGIC %run ./_env.py

# COMMAND ----------

# MAGIC %run ./utils.py

# COMMAND ----------

# MAGIC %run ./service_profile.py

# COMMAND ----------

#import proute

# COMMAND ----------

# s3 bucket to use
bucket = S3Bucket("bus-v2-data", AWS_ACCESS_KEY, AWS_SECRET_KEY)
bucket.allowSpark().mount('s3', ignore_exception=True)

# COMMAND ----------

# util func
ws = lambda path: "/workspace/amit/" + path # return path to my workspace

# resources
EZLINK = bucket.s3("data/ezlink-201702.parquet")
ROUTE = bucket.local("data/lta_scheduled_bus_routes_for_feb2017.csv")

# COMMAND ----------

from datetime import datetime, time

route_schema = dict(service="service", 
                    direction="direction", 
                    stop_code="BusStopCode", 
                    seq="BusStopSequence", 
                    km="km", 
                    dt_from="dt_from", 
                    dt_to="dt_to",
                    time_format='%d/%m/%Y')

ezlink_schema = dict(src="BOARDING_STOP_STN",
                     dst="ALIGHTING_STOP_STN",
                     year="Year",
                     bus_id="BUS_REG_NUM",
                     trip_id="Bus_Trip_Num",
                     journey_id="JOURNEY_ID",
                     travel_mode="TRAVEL_MODE",
                     service="Srvc_Number",
                     direction="Direction",
                     km="Ride_Distance",
                     tap_in_time="tap_in_time",
                     tap_out_time="tap_out_time")

route_valid_for_date = datetime(2017, 2, 1)
days_of_interest = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
am_peak = dict(start_time=time(7, 30), end_time=time(9, 30)) 
pm_peak = dict(start_time=time(17, 0), end_time=time(20, 0))

##TO DO - Separate trunk services that are loop vs non-loop & perform clustering separately
trunk_services = ["2", "3", "5", "7", "8", "9", "10", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "38", "39", "40", "41", "43", "45", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "85", "86", "87", "88", "89", "90", "93", "97", "98", "99", "100", "101", "102", "103", "105", "106", "107", "109", "110", "111", "112", "113", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "128", "129", "130", "131", "132", "133", "134", "135", "136", "137", "138", "139", "141", "143", "145", "147", "151", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "165", "166", "167", "168", "169", "170", "171", "172", "173", "174", "175", "176", "178", "180", "182", "183", "184", "185", "186", "187", "188", "189", "190", "192", "193", "195", "196", "197", "198", "200", "201", "246", "247", "248", "249", "251", "252", "254", "257", "258", "700", "851", "852", "853", "854", "855", "856", "857", "858", "859", "860", "925", "926", "927", "950", "960", "961", "962", "963", "965", "966", "969", "970", "972", "974", "975", "980", "981", "985"] 
# Trunk services shorter than 12 KM and less than 20 bus-stops have been removed from the set above (validated against TransitLink website)

# COMMAND ----------

# route data
route = (Route.from_csv(ROUTE, **route_schema)
         .valid_for(route_valid_for_date))

# ezlink data
ezlink_data = spark.read.parquet(EZLINK)
# Subset bus data
ezlink_data.createOrReplaceTempView('data_table')
ezlink_bus_data = sqlContext.sql('select * from data_table where TRAVEL_MODE="Bus"')
ezlink = Ezlink(ezlink_bus_data, **ezlink_schema)

# Obtain trunk bus services in data that match trunk services list from above & with valid direction
ezlink_bus_srvc = ezlink.dataframe.select('Srvc_Number','Direction').distinct().rdd.map(lambda r:(r[0],r[1])).collect()
trunk_services = list(set([item[0] for item in ezlink_bus_srvc]).intersection(trunk_services))
ezlink_bus_srvc = [item for item in ezlink_bus_srvc if (item[0] in trunk_services) and (item[1] is not None)]

# COMMAND ----------

import numpy as np
from sklearn.preprocessing import normalize

data_frames = []
for item in ezlink_bus_srvc:
  service_of_interest = dict(service=item[0], direction=item[1])
  service_route = route.for_service(**service_of_interest)
  if not service_route.dataframe.empty:
    joined, route_df = (ezlink
                        .for_service(**service_of_interest)
                        .in_days_of_week(days_of_interest)
                        .within_time_range(**am_peak)
                        .get_trips_amit(service_route))
    if not joined.empty:
      # Generate 2D Matrix of origin-destination bus stops for the service (to include even 0 occurrence pairs)
      idx = list(route_df.index)
      mat_sz = len(idx)
      if (idx[0] == idx[mat_sz-1]): # checking for loop service & renaming first and last stops to prevent confusion in mapping
        joined.loc[(joined['source'] == idx[0]),'source'] = idx[0]+"_O"
        idx[0] = idx[0]+"_O"
        joined.loc[(joined['destination'] == idx[mat_sz-1]),'destination'] = idx[mat_sz-1] + "_D"
        idx[mat_sz-1] = idx[mat_sz-1] + "_D"
      
      I = pd.Index(idx,
                   name="")
      C = pd.Index(idx,
                   name="")
      route_df_mat = pd.DataFrame(pd.np.zeros((mat_sz,mat_sz),
                                              dtype=np.int),
                                  index=I,
                                  columns=C)
      
      # Based on counts in joined, add the counts to the matrix
      for i in range(0,joined.shape[0]):
        route_df_mat.loc[joined.iloc[i,1],joined.iloc[i,0]] += joined.iloc[i,2]
      route_df_mat["boarding total"] = route_df_mat.sum(axis=1)
      route_df_mat.loc["alighting total"] = route_df_mat.sum(axis=0)

      # Create "distance-series" (analagous to time series data) features with ordered bus stops in sequence and passenger flow at each bus stop (first half of feature indicates all the boarding while the second half 
      # indicates all the alighting)
      route_dist_series = pd.DataFrame(
        {"service": np.repeat(item[0], 2*mat_sz),
         "direction" : np.repeat(item[1], 2*mat_sz),
         "bus stop code": np.tile(idx,2),
         "Feature": pd.np.zeros(2*mat_sz,dtype=np.int)
        }
      )
      feature1 = route_df_mat["boarding total"].values.astype(float)
      feature1 = np.delete(feature1, mat_sz)
      feature2 = route_df_mat.loc["alighting total"].values.astype(float)
      feature2 = np.delete(feature2, mat_sz)
      feature = np.concatenate((feature1, feature2), axis=0)
      # Normalize passenger flow feature by l2-norm
      feature = normalize(feature.reshape(-1,1), axis=0)   
      route_dist_series['Feature'] = feature
      data_frames.append(route_dist_series)
    
route_dist_series_all = pd.concat(data_frames)

# COMMAND ----------

# Convert Pandas DataFrame to Spark DataFrame
df = spark.createDataFrame(route_dist_series_all)
display(df)

# COMMAND ----------

# Save Spark DataFrame to S3
df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('s3a://bus-v2-data/workspace/amit/bus_dist_series_data_0.1.csv',mode="overwrite")
