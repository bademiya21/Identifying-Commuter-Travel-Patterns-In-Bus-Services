from service_profile import service_profile as sp
import numpy as np
from datetime import datetime, time
from collections import defaultdict
import pandas as pd
import concurrent.futures as cf

# resources and variable initialization
EZLINK = "ezlink-201702-bus.csv"
ROUTE = "lta_scheduled_bus_routes_for_feb2017.csv"

# Schema for all data
route_schema = dict(service="service",
                    direction="direction",
                    stop_code="BusStopCode",
                    seq="BusStopSequence",
                    km="km",
                    dt_from="dt_from",
                    dt_to="dt_to",
                    time_format='%d/%m/%Y')

route_valid_for_date = datetime(2017, 2, 1)
days_of_interest = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
am_peak = dict(start_time=time(7, 30), end_time=time(9, 30))
pm_peak = dict(start_time=time(17, 0), end_time=time(20, 0))

# read data
route = (sp.Route.from_csv(ROUTE, **route_schema)
         .valid_for(route_valid_for_date))

# read ezlink data
col_names = pd.read_csv(EZLINK, nrows=0).columns
ezlink_schema = {'ALIGHTING_STOP_STN': str,
              'BOARDING_STOP_STN': str,
              'BUS_REG_NUM': str,
              'Bus_Trip_Num': str,
              'Direction': np.int64,
              'JOURNEY_ID': np.int64,
              'Ride_Distance': np.float64,
              'TRAVEL_MODE': str,
              'Year': np.int64,
              'tap_in_time': str,
              'tap_out_time': str,
              'Srvc_Number': str,
              'Date': str}
ezlink = pd.read_csv(EZLINK, dtype=ezlink_schema)

# Get list of dates from data
dates = ezlink['Date'].unique()

# Obtain trunk bus services in data that match trunk services list from above & with valid direction
ezlink_bus_srvc = ezlink[['Srvc_Number','Direction']].drop_duplicates().apply(tuple, axis=1).tolist()
ezlink_bus_srvc = [item for item in ezlink_bus_srvc if (item[0] is not None) and (item[1] is not None)]
    
def get_od_pairs_by_service(trips, service_route):
    # column names
    stop_code = service_route.col("stop_code")
    seq = service_route.col("seq")
    
    trips['pax'] = 1  
    ods = trips.groupby(['BOARDING_STOP_STN', 'ALIGHTING_STOP_STN']).agg({'pax':'sum'}).reset_index()
    ods.columns = ["source", "destination", "pax"]
      
    # find out the sequence for source and destination
    route_svc = (service_route
             .dataframe
             .set_index(stop_code)[[seq]])
        
    od_pairs = (ods.set_index("source")
             .join(route_svc, how="left"))
    od_pairs.columns = ["destination", "pax", "source_seq"]
    # workaround for bug where index name disappears after join
    od_pairs.index = od_pairs.index.rename("source") 
    od_pairs = od_pairs.reset_index()
        
    # find seq for destination
    od_pairs = (od_pairs.set_index("destination")
             .join(route_svc, how="left"))
    od_pairs.columns = ["source", "pax", "source_seq", "destination_seq"]
    # workaround for bug where index name disappears after join
    od_pairs.index = od_pairs.index.rename("destination") 
    od_pairs = od_pairs.reset_index()
        
    # find source-destination with smallest number of stop travelled
    od_pairs["stops_travelled"] = od_pairs["destination_seq"] - od_pairs["source_seq"]
    od_pairs = od_pairs.loc[od_pairs["stops_travelled"] > 0]
    od_pairs = (od_pairs.loc[od_pairs.groupby(["source", 
                                         "destination", 
                                         "source_seq", 
                                         "destination_seq"])["stops_travelled"].idxmin()])
    
    return od_pairs, route_svc

def count_bus(index_list, trips, date_list):
     # count number of buses in service for each day at each stop and sum up for all days in dataset of interest
    bus_count = pd.np.zeros(len(index_list),dtype=np.int)
    for date in date_list:
        tmp = trips[(trips['Date'] == date)]
        for i,stop in enumerate(index_list):
            df1 = tmp[tmp['BOARDING_STOP_STN'] == stop]
            df2 = tmp[tmp['ALIGHTING_STOP_STN'] == stop]
            bus_count[i] = bus_count[i] + max(df1[['BUS_REG_NUM','Bus_Trip_Num']].drop_duplicates().shape[0],df2[['BUS_REG_NUM','Bus_Trip_Num']].drop_duplicates().shape[0])

    # Find mean number of buses in the period
    bus_count = np.mean(bus_count)
    
    return bus_count

def fix_dup_entries(dup_entries, od_pairs, trips, index_list):
    for key, value in dup_entries.items():
        if len(value)==2:
            od_pairs.loc[(od_pairs['source'] == key),'source'] = key+"_O"
            od_pairs.loc[(od_pairs['destination'] == key),'destination'] = key + "_D"
            trips['BOARDING_STOP_STN'] = trips['BOARDING_STOP_STN'].where(trips['BOARDING_STOP_STN'] != key,key+"_O")
            trips['ALIGHTING_STOP_STN'] = trips['ALIGHTING_STOP_STN'].where(trips['ALIGHTING_STOP_STN'] != key,key + "_D")
            index_list[value[0]] = index_list[value[0]]+"_O"
            index_list[value[1]] = index_list[value[1]] + "_D"
        else:
            od_pairs.loc[(od_pairs['source_seq'] == (value[0]+1)),'source'] = key+"_O"
            od_pairs.loc[(od_pairs['source_seq'] == (value[1]+1)),'source'] = key+"_I"

            indices = trips.index[trips['BOARDING_STOP_STN'] == key].tolist()

            for i in indices:
                location = index_list.index(trips.loc[i,'ALIGHTING_STOP_STN'])
                if (location <= value[1]):
                    trips.loc[i,'BOARDING_STOP_STN'] = key+"_O"
                else:
                    trips.loc[i,'BOARDING_STOP_STN'] = key+"_I"

            index_list[value[0]] = index_list[value[0]]+"_O"
            index_list[value[1]] = index_list[value[1]] + "_I"

            od_pairs.loc[(od_pairs['destination_seq'] == (value[1]+1)),'destination'] = key + "_I"
            od_pairs.loc[(od_pairs['destination_seq'] == (value[2]+1)),'destination'] = key + "_D"

            indices = trips.index[trips['ALIGHTING_STOP_STN'] == key].tolist()

            for i in indices:
                location = index_list.index(trips.loc[i,'BOARDING_STOP_STN'])
                if (location <= value[1]):
                    trips.loc[i,'ALIGHTING_STOP_STN'] = key+"_I"
                else:
                    trips.loc[i,'ALIGHTING_STOP_STN'] = key+"_D"

            index_list[value[2]] = index_list[value[2]] + "_D"
            
    return od_pairs, trips, index_list
    

def _helper(x):
    service_of_interest = dict(service=x[0], direction=x[1])
    service_route = route.for_service(**service_of_interest)
    if not service_route.dataframe.empty:
        ezlink_sub = ezlink[(ezlink['Srvc_Number']==x[0]) & (ezlink['Direction']==x[1])]
        if (len(ezlink_sub.head(1)) > 0):
            joined, route_df = get_od_pairs_by_service(ezlink_sub, service_route)
            return cumulative_passengers_service(x, ezlink_sub, joined, route_df)
    return pd.DataFrame(columns=["service","direction","bus stop code","seq","bus count","Net Passengers on Bus at BusStop"])
    
def cumulative_passengers_service(bus_srvc, trips, od_pairs, route):
    
    idx = list(route.index)
    mat_sz = len(idx)
    
    # Check for bus calling at the same bus stop during different segments of the trip (loop services) & rename bus stops accordingly
    dup_entries = defaultdict(list)
    for i,entry in enumerate(idx):
        dup_entries[entry].append(i)
    
    dup_entries = {k:v for k,v in dup_entries.items() if len(v)>1}
    
    if (dup_entries):
        od_pairs, trips, idx = fix_dup_entries(dup_entries, od_pairs, trips, idx)
                    
    bus_count = count_bus(idx, trips, dates)
    
    # Generate 2D Matrix of origin-destination bus stops for the service (to include even 0 occurrence pairs)
    I = pd.Index(idx,
                 name="")
    C = pd.Index(idx,
                 name="")
    route_df_mat = pd.DataFrame(pd.np.zeros((mat_sz,mat_sz),
                                            dtype=np.int),
                                index=I,
                                columns=C)

    # Based on counts in joined, add the counts to the matrix
    for i in range(0,od_pairs.shape[0]):
      route_df_mat.loc[od_pairs.iloc[i,1],od_pairs.iloc[i,0]] += od_pairs.iloc[i,2]
    boarding_total = route_df_mat.sum(axis=1).values
    alighting_total = route_df_mat.sum(axis=0).values

    # Create cumulative net passengers on board each service
    route_dist_series = pd.DataFrame(
      {"service": np.repeat(bus_srvc[0], mat_sz),
       "direction" : np.repeat(bus_srvc[1], mat_sz),
       "bus stop code": idx,
       "seq": list(range(mat_sz)),
       "bus count": np.repeat(bus_count,mat_sz),
       "Net Passengers on Bus at BusStop": pd.np.zeros(mat_sz,dtype=np.int)
      }
    )

    route_dist_series.loc[0,'Net Passengers on Bus at BusStop'] = boarding_total[0]
    for j in range(1,route_dist_series.shape[0]):
        route_dist_series.loc[j,'Net Passengers on Bus at BusStop'] = route_dist_series.loc[j-1,'Net Passengers on Bus at BusStop'] + (boarding_total[j] - alighting_total[j])

    return route_dist_series

if __name__ == '__main__':
    #data_frames = []
    ### Create a pool of processes. By default, one is created for each CPU in your machine.
    with cf.ProcessPoolExecutor() as executor:
        ### Process the list of bus services, but split the work across the process pool to use all CPUs
        data_frames = list(executor.map(_helper, ezlink_bus_srvc))
        #for n in executor.map(_helper, ezlink_bus_srvc):
           #data_frames.append(n)
    route_dist_series_all = pd.concat(data_frames)
    route_dist_series_all.to_csv("cumulative_net_commuter_on_bus_cf.csv", index = False)