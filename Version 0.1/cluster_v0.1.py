# Databricks notebook source
# MAGIC %run ./_env.py

# COMMAND ----------

# MAGIC %run ./utils.py

# COMMAND ----------

# MAGIC %run ./service_profile.py

# COMMAND ----------

#import libraries
#import proute
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# COMMAND ----------

# s3 bucket to use
bucket = S3Bucket("bus-v2-data", AWS_ACCESS_KEY, AWS_SECRET_KEY)
bucket.allowSpark().mount('s3', ignore_exception=True)

# resources
path = bucket.s3("workspace/amit/bus_dist_series_data_0.1.csv")
#df = sqlContext.read.format('csv').options(header='true', inferSchema='true').load(path)
schema = StructType([
  StructField('Feature', DoubleType()),
  StructField('bus stop code', StringType()),
  StructField('direction', IntegerType()),
  StructField('service', StringType()),
])
df = spark.read.csv(path, header = "true", schema = schema)
display(df)

# COMMAND ----------

# Obtain trunk bus services in data and corresponding directions
bus_srvc_direc = df.select('service','direction').distinct().rdd.map(lambda r:(r[0],r[1])).collect()

# Convert spark dataframe to pandas
dist_series_data = df.toPandas()

# COMMAND ----------

# Compute DTW distances between all features and compute a 2D distance matrix for clustering
# Create empty pandas dataframe to store all the distances with the indices labelled by the bus service and direction
mat_sz = len(bus_srvc_direc)
I = pd.Index(bus_srvc_direc,
             name="")
C = pd.Index(bus_srvc_direc,
             name="")  
dtw_distance_mat = pd.DataFrame(pd.np.zeros((mat_sz,mat_sz),
                                              dtype=np.float),
                                  index=I,
                                  columns=C)

# Create feature list for easy indexing
feature_list = []
for item in bus_srvc_direc:
  feature = dist_series_data.loc[(dist_series_data['service'] == item[0]) & (dist_series_data['direction'] == item[1]),'Feature'].values
  feature = feature.reshape((feature.shape[0],))
  feature_list.append(feature)  

# COMMAND ----------

# Compute distance matrix for feature list
for i in range(0,len(feature_list)):
  for j in range(i+1,len(feature_list)):
    #radius = max(1, abs(len(feature_list[i])-len(feature_list[j])))
    distance, path = fastdtw(feature_list[i], feature_list[j], dist=euclidean)
    dtw_distance_mat.loc[(bus_srvc_direc[i][0],bus_srvc_direc[i][1]),(bus_srvc_direc[j][0],bus_srvc_direc[j][1])] = distance
    dtw_distance_mat.loc[(bus_srvc_direc[j][0],bus_srvc_direc[j][1]),(bus_srvc_direc[i][0],bus_srvc_direc[i][1])] = distance

dtw_distance_mat

# COMMAND ----------

# Convert Pandas DataFrame to Spark DataFrame
df = spark.createDataFrame(dtw_distance_mat)

# Save Spark DataFrame to S3
df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('s3a://bus-v2-data/workspace/amit/dtw_distance_mat_0.1.csv',mode="overwrite")

# COMMAND ----------

# s3 bucket to use
bucket = S3Bucket("bus-v2-data", AWS_ACCESS_KEY, AWS_SECRET_KEY)
bucket.allowSpark().mount('s3', ignore_exception=True)

# util func
ws = lambda path: "/workspace/amit/" + path # return path to my workspace

path = bucket.s3("workspace/amit/dtw_distance_mat_0.1.csv")
df = sqlContext.read.format('csv').options(header='true', inferSchema='true').load(path)

# COMMAND ----------

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# COMMAND ----------

# resources
EZLINK = bucket.s3("data/ezlink-201702.parquet")
ROUTE = bucket.local("data/lta_scheduled_bus_routes_for_feb2017.csv")

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

route_valid_for_date = datetime(2017, 2, 14)
days_of_interest = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
am_peak = dict(start_time=time(7, 30), end_time=time(9, 30)) 

# route data
route = (Route.from_csv(ROUTE, **route_schema)
         .valid_for(route_valid_for_date))

# ezlink data
ezlink_data = spark.read.parquet(EZLINK)

# Subset bus data
ezlink_data.createOrReplaceTempView('data_table')
ezlink_bus_data = sqlContext.sql('select * from data_table where TRAVEL_MODE="Bus"')
ezlink = Ezlink(ezlink_bus_data, **ezlink_schema)

# COMMAND ----------

# Convert spark dataframe to pandas
dtw_distance_mat = df.toPandas()
ezlink_bus_srvc = tuple(dtw_distance_mat.columns.values)
idx = dict(enumerate(ezlink_bus_srvc, start=0))
dtw_distance_mat.rename(index = idx,inplace=True)

# COMMAND ----------

dtw_distance_mat_condensed = squareform(dtw_distance_mat)

# COMMAND ----------

# generate the linkage matrix
Z = linkage(dtw_distance_mat_condensed, 'ward')
plt.figure(figsize=(100, 40))
plt.title('Hierarchical Clustering Ward Linkage Dendrogram of Commuter-Bus Travel Patterns', fontsize = 36)
plt.xlabel('sample index', fontsize = 36)
plt.ylabel('distance', fontsize = 36)
dn = dendrogram(Z, labels=dtw_distance_mat.index, leaf_font_size=16.)
display(plt.show())

# COMMAND ----------

plt.ioff()
fig = plt.figure(figsize=(100, 40))
for clust_num in range(5,17):
  nodes = fcluster(Z, clust_num, criterion="maxclust")
  df = pd.DataFrame({"bus service direc" : bus_srvc_direc,
                     "feature" : feature_list,
                     "cluster" : nodes
                     }
      )
  for cluster in range(1,clust_num+1):
    df_sub = df.loc[df["cluster"] == cluster]
    dbutils.fs.mkdirs(bucket.s3("workspace/amit/plots_0.1/Clusters="+str(clust_num)+"/"+ str(cluster)))
    SAVED_PLOTS_LOCATION = bucket.local("workspace/amit/plots_0.1/Clusters="+str(clust_num)+"/"+ str(cluster))
    for ind,feat in enumerate(df_sub["feature"]):
        # Save feature plot of each service and direction for the cluster
        bus_stops = dist_series_data.loc[(dist_series_data['service'] == df_sub.iloc[ind,0][0]) & (dist_series_data['direction'] == df_sub.iloc[ind,0][1]),'bus stop code'].values
        y_pos = np.arange(len(bus_stops))
        plt.bar(y_pos,feat,  align='center', alpha=0.5)
        plt.xticks(y_pos, bus_stops, rotation='vertical')
        plt.title('Passenger Flow vs Stops for Service {} Direction {} in Cluster: {}'.format(df_sub.iloc[ind,0][0],df_sub.iloc[ind,0][1],cluster), fontsize=96)
        plt.xlabel('Bus Stops', fontsize=96)
        plt.ylabel('Normalized Passenger Flow',fontsize=96)
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 64)
        fig.savefig(SAVED_PLOTS_LOCATION+'/'+str(df_sub.iloc[ind,0][0])+'_'+str(df_sub.iloc[ind,0][1])+'.png',bbox_inches='tight')
        plt.clf()
        
        # Save corresponding vega diagram along with feature plot 
        srvc=dict(service=df_sub.iloc[ind,0][0], direction=df_sub.iloc[ind,0][1])
        service_route = route.for_service(**srvc)
        trips = (ezlink
                 .for_service(**srvc)
                 .in_days_of_week(days_of_interest)
                 .within_time_range(**am_peak)
                 .get_trips(service_route))
        edges = trips.to_edges(source="src_seq", target="dst_seq", value="pax")
        nodes = service_route.to_nodes(name="stop_code", index="seq", order="km")
        arc_html = vega(arc_diagram(edges=edges, nodes=nodes), render=False)
        arc_file = open(SAVED_PLOTS_LOCATION+'/'+str(df_sub.iloc[ind,0][0])+'_'+str(df_sub.iloc[ind,0][1])+'_vega_arc.html',"w")
        arc_file.write(arc_html)
        arc_file.close()
