#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:36:25 2021

@author: feng779
"""

import numpy as np
import math
from osgeo import osr
#import ogr
from osgeo import ogr
import pdb

def distance(origin, destination):
    
    """ https://gist.github.com/rochacbruno/2883505 """
    
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def findNearset1D(x1, x2):
    """
    Return the J,I indeces of the nearest grid cell to x,y
    """
    dist = np.abs(x2-x1)
    
    return np.argwhere(dist==dist.min())[0][0]

def findNearset2D(x,y,lon,lat):
    """
    Return the J,I indices of the nearst grid cell to x,y
    """
    dist = np.sqrt( (lon - x)**2 + (lat - y)**2)

    return np.argwhere(dist==dist.min())[0]

def readShpPointLine(shpfile,FIELDNAME=None):
    """ Reads a shapefile with line or point geometry and returns x,y,z
    
    See this tutorial:
        http://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides1.pdf
    """

    # Open the shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp = driver.Open(shpfile, 0)
    #pdb.set_trace()
    #shp = ogr.Open(shpfile)

    lyr = shp.GetLayer()

    lyr.ResetReading()
    XY=[]
    field=[]
    for feat in lyr:
        feat_defn = lyr.GetLayerDefn()
        for i in [range(feat_defn.GetFieldCount())[0]]:
            field_defn = feat_defn.GetFieldDefn(i)
            if FIELDNAME==None:
                geom = feat.GetGeometryRef()
#                ztmp = float(feat.GetField(i))
                ztmp = 0 # placeholder
                if geom.GetGeometryType() == ogr.wkbPoint: # point
                    field.append(ztmp)
                    XY.append([geom.GetX(),geom.GetY()])
                elif geom.GetGeometryType() in [2, 3002]:  # line
                    xyall=geom.GetPoints()
                    XY.append(np.asarray(xyall))
                    #field.append(ztmp)

                elif geom.GetGeometryType() in [5, 3005]:  # multiline
                    for ii in range(0,geom.GetGeometryCount()):
                        geom2 = geom.GetGeometryRef(ii)
                        xyall=geom2.GetPoints()
                        XY.append(np.asarray(xyall))
                        #field.append(ztmp)

            elif field_defn.GetName() == FIELDNAME:
                geom = feat.GetGeometryRef()
                #ztmp = float(feat.GetField(i))
                ztmp = feat.GetField(i)
                if geom.GetGeometryType() == ogr.wkbPoint: # point
                    field.append(ztmp)
                    #XY.append([geom.getX(),geom.getY()])
                    XY.append(geom.GetPoints())
                elif geom.GetGeometryType() == 2:  # line
                    xyall=geom.GetPoints()
                    XY.append(np.asarray(xyall))
                    field.append(ztmp)

                elif geom.GetGeometryType() == 5:  # multiline
                    for ii in range(0,geom.GetGeometryCount()):
                        geom2 = geom.GetGeometryRef(ii)
                        xyall=geom2.GetPoints()
                        XY.append(np.asarray(xyall))
                        field.append(ztmp)

    shp=None

    return XY,field

def readShpPoly(shpfile,FIELDNAME = None):
    """ Reads a shapefile with polygon geometry and returns x,y and FIELDNAME value
    
    See this tutorial:
        http://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides1.pdf
    """
    # Open the shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    shp = driver.Open(shpfile, 0)
    
    lyr = shp.GetLayer()
    
    lyr.ResetReading()
    XY=[]
    field=[]
    for feat in lyr:
        feat_defn = lyr.GetLayerDefn()
        for i in range(feat_defn.GetFieldCount()):
            field_defn = feat_defn.GetFieldDefn(i)
            if FIELDNAME==None:
                # Get all of the polygons
                geom = feat.GetGeometryRef()
                ztmp = feat.GetField(i)
               
                if geom.GetGeometryType() == ogr.wkbPolygon:  # Polygon
                
                    for ii in range(0,geom.GetGeometryCount()):
                        geom2 = geom.GetGeometryRef(ii)
                        xyall=geom2.GetPoints()
                        
                        XY.append(np.asarray(xyall))
                        
                if geom.GetGeometryType() == ogr.wkbMultiPolygon:  # Multi Polygon
                    for ii in range(0,geom.GetGeometryCount()):
                        geom2 = geom.GetGeometryRef(ii)
                        for jj in range(0,geom2.GetGeometryCount()):
                            geom3 = geom2.GetGeometryRef(jj)
                            xyall=geom3.GetPoints()
                            
                            XY.append(np.asarray(xyall))
                
                
            
            elif field_defn.GetName() == FIELDNAME:
                geom = feat.GetGeometryRef()
                ztmp = feat.GetField(i)
               
                if geom.GetGeometryType() == ogr.wkbPolygon:  # Polygon
                
                    for ii in range(0,geom.GetGeometryCount()):
                        geom2 = geom.GetGeometryRef(ii)
                        xyall=geom2.GetPoints()
                        
                        XY.append(np.asarray(xyall))
                        field.append(ztmp)
                        
                if geom.GetGeometryType() == ogr.wkbMultiPolygon:  # Multi Polygon
                    for ii in range(0,geom.GetGeometryCount()):
                        geom2 = geom.GetGeometryRef(ii)
                        for jj in range(0,geom2.GetGeometryCount()):
                            geom3 = geom2.GetGeometryRef(jj)
                            xyall=geom3.GetPoints()
                            
                            XY.append(np.asarray(xyall))
                            field.append(ztmp)  
    
    shp=None
    return XY,field
