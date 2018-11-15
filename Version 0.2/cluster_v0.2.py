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
path = bucket.s3("workspace/amit/bus_dist_series_data_0.2.csv")
schema = StructType([
  StructField('Feature', DoubleType()),
  StructField('direction', IntegerType()),
  StructField('service', StringType()),
])
df = spark.read.csv(path, header = "true", schema = schema)
display(df)

# resources
path = bucket.s3("workspace/amit/bus_dist_series_data_loop_0.2.csv")
schema = StructType([
  StructField('Feature', DoubleType()),
  StructField('direction', IntegerType()),
  StructField('service', StringType()),
])
df_loop = spark.read.csv(path, header = "true", schema = schema)
display(df_loop)

# COMMAND ----------

# Obtain trunk bus services in data and corresponding directions
bus_srvc_direc = df.select('service','direction').distinct().rdd.map(lambda r:(r[0],r[1])).collect()
bus_srvc_direc_loop = df_loop.select('service','direction').distinct().rdd.map(lambda r:(r[0],r[1])).collect()

# Convert spark dataframe to pandas
dist_series_data = df.toPandas()
dist_series_data_loop = df_loop.toPandas()

# COMMAND ----------

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
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
  feature = feature.reshape(1,-1)
  #feature = feature/np.amax(feature)
  feature_list.append(feature)
feature_list = np.concatenate(feature_list, axis=0)

#feature_list = MinMaxScaler().fit_transform(feature_list)
#feature_list = StandardScaler().fit_transform(feature_list)
#feature_list = Normalizer(norm='l2').fit_transform(feature_list)

# Create empty pandas dataframe to store all the distances with the indices labelled by the bus service and direction
mat_sz = len(bus_srvc_direc_loop)
I = pd.Index(bus_srvc_direc_loop,
             name="")
C = pd.Index(bus_srvc_direc_loop,
             name="")  
dtw_distance_mat = pd.DataFrame(pd.np.zeros((mat_sz,mat_sz),
                                              dtype=np.float),
                                  index=I,
                                  columns=C)

# Create feature list for easy indexing
feature_list_loop = []
for item in bus_srvc_direc_loop:
  feature = dist_series_data_loop.loc[(dist_series_data_loop['service'] == item[0]) & (dist_series_data_loop['direction'] == item[1]),'Feature'].values
  feature = feature.reshape(1,-1)
  #feature = feature/np.amax(feature)
  feature_list_loop.append(feature)
feature_list_loop = np.concatenate(feature_list_loop, axis=0)

#feature_list_loop = MinMaxScaler().fit_transform(feature_list_loop)
#feature_list_loop = StandardScaler().fit_transform(feature_list_loop)
#feature_list_loop = Normalizer(norm='l2').fit_transform(feature_list_loop)

# COMMAND ----------

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colors
import six

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
ezlink = (ezlink
          .in_days_of_week(days_of_interest)
          .within_time_range(**am_peak))
ezlink.dataframe.cache()

# COMMAND ----------

for clust_num in range(2,41):
  labels = KMeans(n_clusters=clust_num, init='random', max_iter=150000, n_init=1000, random_state=1, n_jobs = -1).fit_predict(feature_list)
  labels_loop = KMeans(n_clusters=clust_num, init='random', max_iter=150000, n_init=1000, random_state=1, n_jobs = -1).fit_predict(feature_list_loop)
  # The silhouette_score gives the average value for all the samples.
  # This gives a perspective into the density and separation of the formed
  # clusters
  silhouette_avg = silhouette_score(feature_list, labels)
  print("For n_clusters =", clust_num,
        "The average silhouette_score for non-loop services is :", silhouette_avg)
  silhouette_avg = silhouette_score(feature_list_loop, labels_loop)
  print("For n_clusters =", clust_num,
        "The average silhouette_score for loop services is :", silhouette_avg)

# COMMAND ----------

clust_num = 17
labels = KMeans(n_clusters=clust_num, init='random', max_iter=150000, n_init=1000, random_state=1, n_jobs = -1).fit_predict(feature_list)
clust_num_loop = 11
labels_loop = KMeans(n_clusters=clust_num_loop, init='random', max_iter=150000, n_init=1000, random_state=1, n_jobs = -1).fit_predict(feature_list_loop)

from mpl_toolkits.mplot3d import Axes3D
import itertools

pca1 = PCA(n_components=3)
feature_list_r = pca1.fit(feature_list).transform(feature_list)
pca2 = PCA(n_components=3)
feature_list_loop_r = pca2.fit(feature_list_loop).transform(feature_list_loop)

markers=itertools.cycle(('.',',','o','v','^','<','>','1','2','3','4','8','s','p','*','h','H','+','x','d','|','_'))
colors_ = list(six.iteritems(colors.cnames))

fig = plt.figure(figsize=(16, 16))

ax = fig.add_subplot(221, projection='3d')

for i in range(0,clust_num):
    ax.scatter(feature_list_r[labels==i,0],feature_list_r[labels==i,1],feature_list_r[labels==i,2],c=colors_[i][0], marker=next(markers), s = 64)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

ax.set_title('PCA of Features for Non-Loop Services')

ax = fig.add_subplot(222, projection='3d')

for i in range(0,clust_num_loop):
   ax.scatter(feature_list_loop_r[labels_loop==i,0],feature_list_loop_r[labels_loop==i,1],feature_list_loop_r[labels_loop==i,2],c=colors_[i][0], marker=next(markers), s = 64)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

ax.set_title('PCA of Features for Loop Services')

ax = plt.subplot(2, 2, 3)

for i in range(0,clust_num):
    plt.scatter(feature_list_r[labels==i,0],feature_list_r[labels==i,1],c=colors_[i][0], marker=next(markers), s = 64)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')

ax.set_title('PCA of Features for Non-Loop Services')

ax = plt.subplot(2, 2, 4)

for i in range(0,clust_num_loop):
    plt.scatter(feature_list_loop_r[labels_loop==i,0],feature_list_loop_r[labels_loop==i,1],c=colors_[i][0], marker=next(markers), s = 64)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')

ax.set_title('PCA of Features for Loop Services')

display(plt.show())

# COMMAND ----------

plt.ioff()

for clust_num in range(25,31):
  labels = KMeans(n_clusters=clust_num, init='random', max_iter=150000, n_init=1000, random_state=1, n_jobs = -1).fit_predict(feature_list)
  df = pd.DataFrame({"bus service direc" : bus_srvc_direc,
                     "feature" : list(feature_list),
                     "cluster" : labels.reshape(-1,)
                     }
      )

  for cluster in range(0,clust_num):
    df_sub = df.loc[df["cluster"] == cluster]
    dbutils.fs.mkdirs(bucket.s3("workspace/amit/plots_0.2/KMeans/Non-Loop/Clusters="+str(clust_num)+"/"+ str(cluster)))
    SAVED_PLOTS_LOCATION = bucket.local("workspace/amit/plots_0.2/KMeans/Non-Loop/Clusters="+str(clust_num)+"/"+ str(cluster))

    for ind,feat in enumerate(df_sub["feature"]):
      fig = plt.figure(figsize=(100, 50))
      # Save feature plot of each service and direction for the cluster
      y_pos = np.arange(20)
      plt.bar(y_pos, feat, alpha=0.5)
      plt.title('Passenger Flow vs Normalized Distance for Service {} Direction {} in Cluster: {}'.format(df_sub.iloc[ind,0][0],df_sub.iloc[ind,0][1],cluster), fontsize=96)
      plt.xlabel('Normalized Distance (1 km)', fontsize=96)
      plt.ylabel('Normalized Passenger Flow',fontsize=96)
      ax = plt.gca()
      ax.tick_params(axis = 'both', which = 'major', labelsize = 64)
      fig.savefig(SAVED_PLOTS_LOCATION+'/'+str(df_sub.iloc[ind,0][0])+'_'+str(df_sub.iloc[ind,0][1])+'.png',bbox_inches='tight')
      plt.close()

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


for clust_num_loop in range(6,31):
  labels_loop = KMeans(n_clusters=clust_num_loop, init='random', max_iter=150000, n_init=1000, random_state=1, n_jobs = -1).fit_predict(feature_list_loop)
  df_loop = pd.DataFrame({"bus service direc" : bus_srvc_direc_loop,
                     "feature" : list(feature_list_loop),
                     "cluster" : labels_loop.reshape(-1,)
                     }
      )
  for cluster in range(0,clust_num_loop):
    df_sub_loop = df_loop.loc[df_loop["cluster"] == cluster]    
    dbutils.fs.mkdirs(bucket.s3("workspace/amit/plots_0.2/KMeans/Loop/Clusters="+str(clust_num_loop)+"/"+ str(cluster)))
    SAVED_PLOTS_LOCATION_LOOP = bucket.local("workspace/amit/plots_0.2/KMeans/Loop/Clusters="+str(clust_num_loop)+"/"+ str(cluster))

    for ind,feat in enumerate(df_sub_loop["feature"]):
      fig = plt.figure(figsize=(100, 50))
      # Save feature plot of each service and direction for the cluster
      y_pos = np.arange(20)
      plt.bar(y_pos, feat, alpha=0.5)
      plt.title('Passenger Flow vs Normalized Distance for Service {} Direction {} in Cluster: {}'.format(df_sub_loop.iloc[ind,0][0],df_sub_loop.iloc[ind,0][1],cluster), fontsize=96)
      plt.xlabel('Normalized Distance (1 km)', fontsize=96)
      plt.ylabel('Normalized Passenger Flow',fontsize=96)
      ax = plt.gca()
      ax.tick_params(axis = 'both', which = 'major', labelsize = 64)
      fig.savefig(SAVED_PLOTS_LOCATION_LOOP+'/'+str(df_sub_loop.iloc[ind,0][0])+'_'+str(df_sub_loop.iloc[ind,0][1])+'.png',bbox_inches='tight')
      plt.close()

      # Save corresponding vega diagram along with feature plot 
      srvc=dict(service=df_sub_loop.iloc[ind,0][0], direction=df_sub_loop.iloc[ind,0][1])
      service_route = route.for_service(**srvc)
      trips = (ezlink
               .for_service(**srvc)
               .get_trips(service_route))
      edges = trips.to_edges(source="src_seq", target="dst_seq", value="pax")
      nodes = service_route.to_nodes(name="stop_code", index="seq", order="km")
      arc_html = vega(arc_diagram(edges=edges, nodes=nodes), render=False)
      arc_file = open(SAVED_PLOTS_LOCATION_LOOP+'/'+str(df_sub_loop.iloc[ind,0][0])+'_'+str(df_sub_loop.iloc[ind,0][1])+'_vega_arc.html',"w")
      arc_file.write(arc_html)
      arc_file.close()
