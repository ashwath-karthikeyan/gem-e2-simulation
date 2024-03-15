#!/usr/bin/env python
import os
import csv
from math import *
import numpy as np
import rospy
from novatel_gps_msgs.msg import Inspva  # Import the Inspva message type

olat = 40.0928563
olon = -88.2359994

waypoints = []

gps_values = []

def callback(data):
    gps_lats = data.latitude
    gps_longs = data.longitude
    gps_heading = data.azimuth

    gps_values.append([gps_lats, gps_longs, gps_heading])

    val = ll2xy(gps_lats, gps_longs, olat, olon)
    wp_lats = val[0]
    wp_longs = val[1]

    waypoints.append([wp_lats, wp_longs, gps_heading])

def append_to_csv(file_name, data):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def listener():
    rospy.init_node('inspva_listener', anonymous=True)
    rospy.Subscriber("/novatel/inspva", Inspva, callback)

    rospy.spin()

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

def ll2xy(lat, lon, orglat, orglon):
    '''
    AlvinXY: Lat/Long to X/Y
    Converts Lat/Lon (WGS84) to Alvin XYs using a Mercator projection.

    Args:
      lat (float): Latitude of location
      lon (float): Longitude of location
      orglat (float): Latitude of origin location
      orglon (float): Longitude of origin location

    Returns:
      tuple: (x,y) where...
        x is Easting in m (Alvin local grid)
        y is Northing in m (Alvin local grid)
    '''
    x = (lon - orglon) * mdeglon(orglat)
    y = (lat - orglat) * mdeglat(orglat)
    return (x,y)


if __name__ == '__main__':
    listener()
    append_to_csv('waypts_new.csv', waypoints)
    append_to_csv('gps_values_new.csv', gps_values)