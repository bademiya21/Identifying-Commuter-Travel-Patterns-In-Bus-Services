# Identifying Commuter Travel Patterns In Bus Services
A project I did with Land Transport Authority, a statutory board, whose main role is to manage the transportation infra of Singapore which includes public transport like bus and trains. The agency was interested to understand how the bus services were being utilized by commuters during peak hours and if interventions could be introduced to further enhance commuter experience on bus services e.g. shorter waiting time, faster trips with skipping of bus stops etc. This required understanding archetypes of travel patterns by commuters in bus services. This project is an extension of what was previously done here: https://blog.data.gov.sg/fingerprint-of-a-bus-route-73e5be53dcf0

The travel data provided by the agency consisted of daily EZ-Link card transactions across all public transport services and EZ-Link machines across the island. Each line of the data consisted of the following fields:

| ALIGHTING_STOP_STN | BOARDING_STOP_STN | BUS_REG_NUM | Bus_Trip_Num | Direction | JOURNEY_ID   | Ride_Distance | TRAVEL_MODE | Year | tap_in_time                   | tap_out_time                  | Srvc_Number | Date       |
|--------------------|-------------------|-------------|--------------|-----------|--------------|---------------|-------------|------|-------------------------------|-------------------------------|-------------|------------|

Data specific to bus services and their routes with distance and bus stop information was also provided. Each line of the data consisted of the following fields:

| service | direction | BusStopCode | BusStopSequence | km   | dt_from    | dt_to      |
|---------|-----------|-------------|-----------------|------|------------|------------|

Using these 2 data sources for information, I worked on creating features that represented commuter travel on bus services and clustered these features accordingly to find archetypes of travel patterns and to group bus services with similar commuter travel patterns. The idea for this is that instead of applying intervention on each bus service, the same intervention could be applied to all the bus services that fall in the cluster. I created 2 approaches which are detailed in the folders, Version 1 and Version 2. The python scripts and the accompanying notebooks explain what and how it was done and the drawbacks (where found) of the approaches.

Alternatively, another approach of targetting the commuter experience was to look at how much the buses were occupied during the peak hours. If the buses are empty for most of the trip, interventions could be introduced like redeploying buses to a busier route or to introduce short working trips for sections of the trip where the bus service is heavily utilized. The work for this can be found in the Service Utilization folder. 

Results and intermediate results/data have not been provided as there are confidentiality concerns.

This work was mainly done in Python (PySpark) using Jupyter Notebook on Databricks and AWS resources (Spark cluster).
