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

trunk_services = ["2", "3", "5", "7", "8", "10", "12", "13", "14", "16", "17", "21", "22", "25", "26", "28", "30", "31", "32", "33", "38", "39", "43", "45", "48", "50", "51", "52", "54", "55", "56", "57", "58", "59", "61",    "65", "66", "67", "70", "72", "74", "75", "76", "77", "80", "85", "86", "87", "88", "93", "97", "99", "100", "103", "105", "106", "107", "109", "117", "118", "123", "124", "128", "129", "130", "131", "132", "133", "136", "137", "139", "141", "143", "145", "147", "151", "153", "154", "155", "156", "157", "159", "161", "162", "163", "165", "166", "167", "168", "169", "170", "171", "172", "174", "175", "176", "178", "185", "186", "187", "188", "190", "192", "193", "196", "197", "198", "700", "851", "852", "853", "854", "855", "856", "925", "960", "961", "963", "969", "970", "974", "980", "981", "985"] 

trunk_services_loop = ["9", "15", "18", "19", "20", "23", "24", "27", "29", "34", "35", "36", "40", "41", "47", "49", "53", "60", "62", "63", "64", "68", "69", "71", "73", "78", "79", "81", "82", "83", "89", "90", "98", "101", "102", "110", "111", "112", "113", "116", "119", "120", "121", "122", "125", "134", "135", "138", "158", "160", "173", "180", "182", "183", "184", "189", "195", "200", "201", "246", "247", "248", "249", "251", "252", "254", "257", "258", "857", "858", "859", "860", "926", "927", "950", "962", "965", "966", "972", "975"]
# Trunk services shorter than 12 KM and less than 20 bus-stops have been removed from both sets above (validated against TransitLink website)

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
ezlink = (ezlink
          .in_days_of_week(days_of_interest)
          .within_time_range(**am_peak))
ezlink.dataframe.cache() # Load dataframe to memory to speed up access during iterations later

# Obtain trunk bus services in data that match trunk services list from above & with valid direction
ezlink_bus_srvc = ezlink.dataframe.select('Srvc_Number','Direction').distinct().rdd.map(lambda r:(r[0],r[1])).collect()
trunk_services = list(set([item[0] for item in ezlink_bus_srvc]).intersection(trunk_services))
trunk_services_loop = list(set([item[0] for item in ezlink_bus_srvc]).intersection(trunk_services_loop))
ezlink_bus_srvc_loop = [item for item in ezlink_bus_srvc if (item[0] in trunk_services_loop) and (item[1] is not None)]
ezlink_bus_srvc = [item for item in ezlink_bus_srvc if (item[0] in trunk_services) and (item[1] is not None)]

# COMMAND ----------

import numpy as np
from sklearn.preprocessing import normalize

hist_bins = 10

data_frames = []
for item in ezlink_bus_srvc:
  service_of_interest = dict(service=item[0], direction=item[1])
  service_route = route.for_service(**service_of_interest)
  if not service_route.dataframe.empty:
    joined, route_df = (ezlink
                        .for_service(**service_of_interest)
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
      
      # Keep only the 1% largest O-D pairs if number of O-D pairs exceed 300 else choose 3 largest
      if (joined.shape[0] > 300):
        joined = joined.nlargest(int(0.01*joined.shape[0]), 'pax')
      else:
        joined = joined.nlargest(3, 'pax')
          
      
      # Based on counts in joined, add the counts to the matrix
      for i in range(0,joined.shape[0]):
        route_df_mat.loc[joined.iloc[i,1],joined.iloc[i,0]] += joined.iloc[i,2]
      route_df_mat["boarding total"] = route_df_mat.sum(axis=1)
      route_df_mat.loc["alighting total"] = route_df_mat.sum(axis=0)

      # Create "distance-series" (analagous to time series data) features with ordered bus stops in sequence and passenger flow at each bus stop (first half of feature indicates all the boarding while the second half 
      # indicates all the alighting)
      service_route.dataframe['km'] = service_route.dataframe['km']/service_route.dataframe['km'].iloc[-1] # Normalize bus travel distance to 1 km and  normalize distance between bus stops
      route_dist_series = pd.DataFrame(
        {"service": np.repeat(item[0], 2*hist_bins),
         "direction" : np.repeat(item[1], 2*hist_bins),
         "Feature": pd.np.zeros(2*hist_bins,dtype=np.int)
        }
      )
      
      # Generate histograms of passengers vs normalized distance (1 km)
      weight = route_df_mat["boarding total"].values.astype(float)
      weight = np.delete(weight, mat_sz)
      hist1, bin_edges = np.histogram(service_route.dataframe['km'].values.astype(float), bins=hist_bins, weights=weight)
      hist1 = hist1/np.amax(hist1)
      weight = route_df_mat.loc["alighting total"].values.astype(float)
      weight = np.delete(weight, mat_sz)
      hist2, bin_edges = np.histogram(service_route.dataframe['km'].values.astype(float), bins=hist_bins, weights=weight)
      hist2 = hist2/np.amax(hist2)
      hist = np.concatenate((hist1, hist2), axis=0)
            
      route_dist_series['Feature'] = hist
      data_frames.append(route_dist_series)
    else:
      print(item)
    
route_dist_series_all = pd.concat(data_frames)

hist_bins = 10

data_frames = []
for item in ezlink_bus_srvc_loop:
  service_of_interest = dict(service=item[0], direction=item[1])
  service_route = route.for_service(**service_of_interest)
  if not service_route.dataframe.empty:
    joined, route_df = (ezlink
                        .for_service(**service_of_interest)
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
      
      # Keep only the 1% largest O-D pairs if number of O-D pairs exceed 300 else choose 3 largest
      if (joined.shape[0] > 300):
        joined = joined.nlargest(int(0.01*joined.shape[0]), 'pax')
      else:
        joined = joined.nlargest(3, 'pax')
      
      # Based on counts in joined, add the counts to the matrix
      for i in range(0,joined.shape[0]):
        route_df_mat.loc[joined.iloc[i,1],joined.iloc[i,0]] += joined.iloc[i,2]
      route_df_mat["boarding total"] = route_df_mat.sum(axis=1)
      route_df_mat.loc["alighting total"] = route_df_mat.sum(axis=0)

      # Create "distance-series" (analagous to time series data) features with ordered bus stops in sequence and passenger flow at each bus stop (first half of feature indicates all the boarding while the second half 
      # indicates all the alighting)
      service_route.dataframe['km'] = service_route.dataframe['km']/service_route.dataframe['km'].iloc[-1] # Normalize bus travel distance to 1 km and  normalize distance between bus stops
      route_dist_series = pd.DataFrame(
        {"service": np.repeat(item[0], 2*hist_bins),
         "direction" : np.repeat(item[1], 2*hist_bins),
         "Feature": pd.np.zeros(2*hist_bins,dtype=np.int)
        }
      )
      
      # Generate histograms of passengers vs normalized distance (1 km)
      weight = route_df_mat["boarding total"].values.astype(float)
      weight = np.delete(weight, mat_sz)
      hist1, bin_edges = np.histogram(service_route.dataframe['km'].values.astype(float), bins=hist_bins, weights=weight)
      hist1 = hist1/np.amax(hist1)
      weight = route_df_mat.loc["alighting total"].values.astype(float)
      weight = np.delete(weight, mat_sz)
      hist2, bin_edges = np.histogram(service_route.dataframe['km'].values.astype(float), bins=hist_bins, weights=weight)
      hist2 = hist2/np.amax(hist2)
      hist = np.concatenate((hist1, hist2), axis=0)
            
      route_dist_series['Feature'] = hist
      data_frames.append(route_dist_series)
    else:
      print(item)
    
route_dist_series_all_loop = pd.concat(data_frames)

# COMMAND ----------

# Convert Pandas DataFrame to Spark DataFrame
df = spark.createDataFrame(route_dist_series_all)
df_loop = spark.createDataFrame(route_dist_series_all_loop)
display(df_loop)

# COMMAND ----------

# Save Spark DataFrame to S3
df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('s3a://bus-v2-data/workspace/amit/bus_dist_series_data_0.2.csv',mode="overwrite")
df_loop.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('s3a://bus-v2-data/workspace/amit/bus_dist_series_data_loop_0.2.csv',mode="overwrite")
