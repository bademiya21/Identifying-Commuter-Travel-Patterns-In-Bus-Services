# Databricks notebook source
"""
Data models for various data used in the analysis.
"""

from typing import Text, List, Dict, Union, Callable, Any
from datetime import datetime, time
from pyspark.sql import functions as F
import pyspark
import pandas as pd


PandasDataFrame = Union[pd.DataFrame, Callable[[], pd.DataFrame]]
SparkDataFrame = Union[pyspark.sql.DataFrame, Callable[[], pyspark.sql.DataFrame]]

# COMMAND ----------

class PandasDataFrameBase(object):
  """Base class for dataframe wrapper classes"""
  def __init__(self, dataframe: PandasDataFrame, **columns):
    self.dataframe = dataframe
    self.columns = columns
    
  def col(self, colname: Text) -> Text:
    """return the actual column name for a col."""
    return self.columns[colname]

# COMMAND ----------

class Route(PandasDataFrameBase):
  """Class to model a dataframe containing route information for bus services."""
  def __init__(self, 
               dataframe: PandasDataFrame,
               service: Text, 
               direction: Text, 
               stop_code: Text, 
               seq: Text, 
               km: Text, 
               dt_from: Text, 
               dt_to: Text) -> None:
    
    columns = dict(service=service,
                   direction=direction,
                   stop_code=stop_code,
                   seq=seq,
                   km=km,
                   dt_from=dt_from,
                   dt_to=dt_to)
    dataframe = dataframe() if callable(dataframe) else dataframe
    super().__init__((dataframe.sort_values(by=[direction, seq])), **columns)
    
    self.service = dataframe[service]
    self.direction = dataframe[direction]
    self.stop_code = dataframe[stop_code]
    self.seq = dataframe[seq]
    self.km = dataframe[km]
    self.dt_from = dataframe[dt_from]
    self.dt_to = dataframe[dt_to]
    
  @staticmethod
  def from_csv(path: Text, 
               service: Text, 
               direction: Text, 
               stop_code: Text, 
               seq: Text, 
               km: Text, 
               dt_from: Text, 
               dt_to: Text,
               time_format: Text='%d/%m/%Y') -> "Route":
    """Load a Route from a csv file."""
    dataframe = pd.read_csv(path,
                            converters={service: lambda s: s.strip()},
                            dtype={stop_code: str},
                            parse_dates=[dt_from, dt_to], 
                            infer_datetime_format=True,
                            date_parser=lambda x: pd.datetime.strptime(x, time_format))
    
    dataframe[stop_code] = dataframe[stop_code].apply(lambda x: x.zfill(5))
    
    return Route(dataframe,
                 service, 
                 direction, 
                 stop_code, 
                 seq, 
                 km, 
                 dt_from, 
                 dt_to)
  
  def valid_for(self, when: datetime) -> "Route":
    """Return a Route that is valid for a particular datetime."""
    predicate = (self.dt_from <= when) & (self.dt_to >= when)
    return Route(self.dataframe.loc[predicate], **self.columns)
  
  def for_service(self, service: Text, direction: Text) -> "Route":
    """Return a Route for the corresponding service and direction."""
    predicate = (self.service == service) & (self.direction == direction)
    return Route(self.dataframe.loc[predicate], **self.columns)
  
  def to_nodes(self, name="stop_code", index="seq", order="km"):
    """Return as a list of nodes"""
    results = (self.dataframe[[self.col(name), 
                               self.col(index),
                               self.col(order)]])
    results.columns = ["name", "index", "order"]
    results["group"] = 1
    return results.to_dict("records")

# COMMAND ----------

class Trips(PandasDataFrameBase):
  """Class to model a pandas dataframe of source-destination for a service."""
  def __init__(self, 
               dataframe: PandasDataFrame, 
               src: Text, 
               src_seq: Text,
               dst: Text, 
               dst_seq: Text,
               pax: Text) -> None:
    self.dataframe = dataframe() if callable(dataframe) else dataframe
    self.columns = dict(src=src, 
                        src_seq=src_seq, 
                        dst=dst, 
                        dst_seq=dst_seq,
                        pax=pax)
  def to_edges(self, source: Text = "src_seq", target: Text = "dst_seq", value: Text = "pax"):
    """Return as a list of edges"""
    records = (self.dataframe[[self.col(source), 
                            self.col(target),
                            self.col(value)]]
            .rename(columns={
              self.col(source): "source",
              self.col(target): "target",
              self.col(value): "value"})
            .to_dict("records"))    
    # stoopid to_dict bug that cannot properly convert numpy.int64 to int
    return [{"source": int(record.get("source")), 
             "target": int(record.get("target")), 
             "value": float(record.get("value"))} 
            for record in records]
    


# COMMAND ----------

class Ezlink(object):
  """Class to model the ezlink data"""
  def __init__(self, 
               dataframe: SparkDataFrame,
               src: Text,
               dst: Text,
               year: Text,
               bus_id: Text,
               trip_id: Text,
               journey_id: Text,
               travel_mode: Text,
               service: Text,
               direction: Text,
               km: Text,
               tap_in_time: Text,
               tap_out_time: Text,
               date: Text,
               meta: List = None
              ) -> None:
    self.dataframe = dataframe() if callable(dataframe) else dataframe
    self.columns = dict(src=src,
                        dst=dst,
                        year=year,
                        bus_id=bus_id,
                        trip_id=trip_id,
                        journey_id=journey_id,
                        travel_mode=travel_mode,
                        service=service,
                        direction=direction,
                        km=km,
                        tap_in_time=tap_in_time,
                        tap_out_time=tap_out_time,
                        date = date,
                        meta = meta if meta else dict())
    self.src = F.col(src)
    self.dst = F.col(dst)
    self.year = F.col(year)
    self.bus_id = F.col(bus_id)
    self.trip_id = F.col(trip_id)
    self.journey_id = F.col(journey_id)
    self.travel_mode = F.col(travel_mode)
    self.service = F.col(service)
    self.direction = F.col(direction)
    self.km = F.col(km)
    self.tap_in_time = F.col(tap_in_time)
    self.tap_out_time = F.col(tap_out_time)
    self.date = F.col(date)
    self.meta = self.columns["meta"]
    
  def cache(self):
    return self.dataframe.cache()
  
  def take(self, n: int):
    return self.dataframe.take(n)
  
  def annotate(self, key: str, value: Any) -> Dict:
    """Update the meta-data for ezlink"""
    self.columns["meta"].update({key: value})
    return self
    
  def in_days_of_week(self, days: List[Text], input_time_column=None) -> "Ezlink":
    """Return an Ezlink object that is representative for the selected days of week."""
    if self.meta.get("in_days_of_week"):
      raise RuntimeError("Ezlink is already in_days_of_week="
                         .format(self.meta.get("in_days_of_week")))
    # if input time is before 3am, the ride date should be 1 day earlier
    # Ride on Monday 2:30am = Sunday night ride
    input_time_column = input_time_column if input_time_column else self.tap_in_time
    dataframe = (self.dataframe
                 .withColumn('dayofweek',
                             F.when(F.date_format(input_time_column, "HH") < 3,
                                    F.date_format(F.date_add(input_time_column, -1), "EEEE"))
                             .otherwise(F.date_format(input_time_column, "EEEE"))))

    dataframe = (dataframe
                 .filter(F.col('dayofweek').isin(days))
                 .drop('dayofweek'))
    return (Ezlink(dataframe, **self.columns)
            .annotate("in_days_of_week", days))

  def for_service(self, service: Text, direction: Text) -> "Ezlink":
    """
    Filter by service and direction.
    """
    if self.meta.get("is_service") == (service, direction):
    #if self.meta.get("is_service"):
      raise RuntimeError("Ezlink is already is_service="
                         .format(self.meta.get("is_service")))
    
    dataframe = (self.dataframe
                 .filter(self.service == service)
                 .filter(self.direction == direction))
    return (Ezlink(dataframe, **self.columns)
            .annotate("is_service", (service, direction)))
  
  def within_time_range(self, start_time: time, end_time: time) -> "Ezlink":
    """Return ezlink where each journey start or/and end within the specified time range."""
    if self.meta.get("within_time_range"):
      raise RuntimeError("Ezlink is already within_time_range="
                         .format(self.meta.get("within_time_range")))
    
    start_time_str = start_time.strftime("%H:%M:%S")
    end_time_str = end_time.strftime("%H:%M:%S")
    predicate_board = (F.date_format(self.tap_in_time, "HH:mm:ss")    
                       .between(start_time_str, end_time_str)) 
    predicate_alight = (F.date_format(self.tap_out_time, "HH:mm:ss")    
                        .between(start_time_str, end_time_str)) 
    dataframe = (self.dataframe
                 .filter(predicate_board | predicate_alight))

    return (Ezlink(dataframe, **self.columns)
            .annotate("within_time_range", (start_time, end_time)))
  
  def get_trips(self, route: Route):
    """Return a list of source-destination"""
    # column names
    stop_code = route.col("stop_code")
    seq = route.col("seq")
    seq_source = route.col("seq") + "_source"
    seq_destination = route.col("seq") + "_destination"
    
    ods = (self.dataframe
           .withColumn("pax", F.lit(1))
           .groupBy(self.src.alias("source"), 
                    self.dst.alias("destination"))
           .agg(F.count("pax").alias("pax"))
           .toPandas()[["source", "destination", "pax"]])
     
    # find out the sequence for source and destination
    route_df = (route
                .dataframe
                .set_index(stop_code)[[seq]])
    joined = (ods.set_index("source")
              .join(route_df, how="left"))
    joined.columns = ["destination", "pax", "source_seq"]
    # workaround for bug where index name disappears after join
    joined.index = joined.index.rename("source") 
    joined = joined.reset_index()

    # find seq for destination
    joined = (joined.set_index("destination")
              .join(route_df, how="left"))
    joined.columns = ["source", "pax", "source_seq", "destination_seq"]
    # workaround for bug where index name disappears after join
    joined.index = joined.index.rename("destination") 
    joined = joined.reset_index()
    
    # find source-destination with smallest number of stop travelled
    joined["stops_travelled"] = joined["destination_seq"] - joined["source_seq"]
    joined = joined.loc[joined["stops_travelled"] > 0]
    joined = (joined.loc[joined.groupby(["source", 
                                         "destination", 
                                         "source_seq", 
                                         "destination_seq"])["stops_travelled"].idxmin()])

    return Trips(joined.drop("stops_travelled", axis=1).astype({"source_seq": int, "destination_seq": int}),
                 src="source", 
                 src_seq="source_seq", 
                 dst="destination", 
                 dst_seq="destination_seq", 
                 pax="pax")
  
  def get_trips_amit(self, route: Route):
    """Return a list of source-destination"""
    # column names
    stop_code = route.col("stop_code")
    seq = route.col("seq")
    seq_source = route.col("seq") + "_source"
    seq_destination = route.col("seq") + "_destination"
    
    ods = (self.dataframe
           .withColumn("pax", F.lit(1))
           .groupBy(self.src.alias("source"), 
                    self.dst.alias("destination"))
           .agg(F.count("pax").alias("pax"))
           .toPandas()[["source", "destination", "pax"]])
     
    # find out the sequence for source and destination
    route_df = (route
                .dataframe
                .set_index(stop_code)[[seq]])
    
    joined = (ods.set_index("source")
              .join(route_df, how="left"))
    joined.columns = ["destination", "pax", "source_seq"]
    # workaround for bug where index name disappears after join
    joined.index = joined.index.rename("source") 
    joined = joined.reset_index()

    # find seq for destination
    joined = (joined.set_index("destination")
              .join(route_df, how="left"))
    joined.columns = ["source", "pax", "source_seq", "destination_seq"]
    # workaround for bug where index name disappears after join
    joined.index = joined.index.rename("destination") 
    joined = joined.reset_index()
    
    # find source-destination with smallest number of stop travelled
    joined["stops_travelled"] = joined["destination_seq"] - joined["source_seq"]
    joined = joined.loc[joined["stops_travelled"] > 0]
    joined = (joined.loc[joined.groupby(["source", 
                                         "destination", 
                                         "source_seq", 
                                         "destination_seq"])["stops_travelled"].idxmin()])  
    
    return joined, route_df

  def tap_in_by_hour(self) -> pyspark.sql.DataFrame:
    """
    Return a dataframe with the number of tap in by hour.
    """
    return (self.dataframe
            .select(F.hour(self.tap_in_time).alias('hr'))
            .groupBy(F.col('hr'))
            .agg(F.count('hr').alias('pax'))
            .sort(F.col('hr')))

  def num_days(self) -> int:
    """
    Number of distinct days in the ezlink.
    """
    if self._num_days:
      return self._num_days
    
    self._num_days = (self.dataframe
                      .select(self.year,
                              F.dayofyear(self.tap_in_time).alias('day'))
                      .distinct()
                      .count())
    return self._num_days

# COMMAND ----------

# Plot arc diagram
    
def arc_diagram(edges: List, nodes: List, width: int = 800, padding: int = 5) -> Dict:
  """Return the vega spec for the arc diagram"""
  return {
    "$schema": "https://vega.github.io/schema/vega/v3.json",
    "width": width,
    "padding": padding,
    "data": [
      {
        "name": "edges",
        "values": edges
      },
      {
        "name": "sourceDegree",
        "source": "edges",
        "transform": [
          {"type": "aggregate", "groupby": ["source"]}
        ]
      },
      {
        "name": "targetDegree",
        "source": "edges",
        "transform": [
          {"type": "aggregate", "groupby": ["target"]}
        ]
      },
      {
        "name": "nodes",
        "values": nodes,
        "transform": [
          {
            "type": "lookup", "from": "sourceDegree", "key": "source",
            "fields": ["index"], "as": ["sourceDegree"],
            "default": {"count": 0}
          },
          {
            "type": "lookup", "from": "targetDegree", "key": "target",
            "fields": ["index"], "as": ["targetDegree"],
            "default": {"count": 0}
          },
          {
            "type": "formula", "as": "degree",
            "expr": "datum.sourceDegree.count + datum.targetDegree.count"
          }
        ]
      }
    ],

    "scales": [
      {
        "name": "position",
        "type": "linear",
        "domain": {"data": "nodes", "field": "order", "sort": True},
        "range": "width"
      },
      {
        "name": "opacity",
        "type": "linear",
        "domain": {"data": "edges", "field": "value"},
        "range": [0.1, 0.5]
      },       
      {
        "name": "value",
        "type": "linear",
        "domain": {"data": "edges", "field": "value"},
        "range": [0.1, 3]
      },      
      {
        "name": "color",
        "type": "ordinal",
        "range": "category",
        "domain": {"data": "nodes", "field": "group"}
      }
    ],

    "marks": [
      {
        "type": "symbol",
        "name": "layout",
        "interactive": False,
        "from": {"data": "nodes"},
        "encode": {
          "enter": {
            "opacity": {"value": 0}
          },
          "update": {
            "x": {"scale": "position", "field": "order"},
            "y": {"value": 0},
            "size": {"field": "degree", "mult": 2, "offset": 10},
            "fill": {"scale": "color", "field": "group"}
          }
        }
      },
      {
        "type": "path",
        "from": {"data": "edges"},
        "encode": {
          "update": {
            "stroke": {"value": "#000"},
            "strokeOpacity": {"scale": "opacity", "field": "value"},
            "strokeWidth": {"scale": "value", "field": "value"}
          }
        },
        "transform": [
          {
            "type": "lookup", "from": "layout", "key": "datum.index",
            "fields": ["datum.source", "datum.target"],
            "as": ["sourceNode", "targetNode"]
          },
          {
            "type": "linkpath",
            "sourceX": {"expr": "min(datum.sourceNode.x, datum.targetNode.x)"},
            "targetX": {"expr": "max(datum.sourceNode.x, datum.targetNode.x)"},
            "sourceY": {"expr": "0"},
            "targetY": {"expr": "0"},
            "shape": "arc"
          }
        ]
      },
      {
        "type": "symbol",
        "from": {"data": "layout"},
        "encode": {
          "update": {
            "x": {"field": "x"},
            "y": {"field": "y"},
            "fill": {"field": "fill"},
            "size": {"field": "size"}
          }
        }
      },
      {
        "type": "text",
        "from": {"data": "nodes"},
        "encode": {
          "update": {
            "x": {"scale": "position", "field": "order"},
            "y": {"value": 7},
            "fontSize": {"value": 9},
            "align": {"value": "right"},
            "baseline": {"value": "middle"},
            "angle": {"value": -90},
            "text": {"field": "name"}
          }
        }
      }
    ]
  }

