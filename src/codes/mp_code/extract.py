import os
import csv




olat = 40.0928563
olon = -88.2359994

lats = []
longs = []

from math import *
import numpy as np

def  mdeglat(lat):
    '''
    Provides meters-per-degree latitude at a given latitude
    
    Args:
      lat (float): latitude

    Returns:
      float: meters-per-degree value
    '''
    latrad = lat*2.0*pi/360.0 

    dy = 111132.09 - 566.05 * cos(2.0*latrad) \
         + 1.20 * cos(4.0*latrad) \
         - 0.002 * cos(6.0*latrad)
    return dy

def mdeglon(lat):
    '''
    Provides meters-per-degree longitude at a given latitude

    Args:
      lat (float): latitude in decimal degrees

    Returns:
      float: meters per degree longitude
    '''
    latrad = lat*2.0*pi/360.0 
    dx = 111415.13 * cos(latrad) \
         - 94.55 * cos(3.0*latrad) \
	+ 0.12 * cos(5.0*latrad)
    return dx

def xy2ll(x, y, orglat, orglon):

    '''
    X/Y to Lat/Lon
    Converts Alvin XYs to Lat/Lon (WGS84) using a Mercator projection.

    Args:
      x (float): Easting in m (Alvin local grid)
      x (float): Northing in m (Alvin local grid)
      orglat (float): Latitude of origin location
      orglon (float): Longitude of origin location

    Returns:
      tuple: (lat,lon) 
    '''
    lon = x/mdeglon(orglat) + orglon
    lat = y/mdeglat(orglat) + orglat

    return (lat, lon)

for i in len(gps_values):
    val = xy2ll(path_points_lon_x[i], path_points_lat_y[i], olat, olon)
    latitude = val[0]
    longitude = val[1]
    lats.append(latitude)
    longs.append(longitude)


