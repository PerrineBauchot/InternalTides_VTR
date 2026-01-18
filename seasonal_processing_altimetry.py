#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pbauchot

This code allows to identify the seasonal cycles in the satellite data, in order to retrieve the summer cycle in one file and the winter cycle in another file. This separation allows to conduct a seasonal analysis based on altimetry.
"""

from os import listdir
from os.path import isfile, join
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt


directory = './ANALYSES_HARMO'
files = [f for f in listdir(directory) if isfile(join(directory,f))]
print(files)
tmp = 15706
pas_saison = 182.5
n = 0

for f in files:
    data = nc.Dataset(join(directory,f))
    time = data.variables['time'][:,:]
    var_sla = data.variables['sla']
    sla = data.variables['sla'][:,:]
    cycles = data.variables['cycles'][:,:]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    points = data.variables['point'][:]
    load = data.variables['load'][:,:]
    tide = data.variables['tide'][:,:]
    time.mask = np.ma.nomask
    sla.mask = np.ma.nomask
    cycles.mask = np.ma.nomask
    lat.mask = np.ma.nomask
    lon.mask = np.ma.nomask
    points.mask = np.ma.nomask
    load.mask = np.ma.nomask
    tide.mask = np.ma.nomask
    
    data.close()

    time_winter = np.zeros(time.shape)
    time_summer = np.zeros(time.shape)
    cycle_winter = np.zeros(cycles.shape)
    cycle_summer = np.zeros(cycles.shape)
    sla_winter = np.zeros(sla.shape)
    sla_summer = np.zeros(sla.shape)
    lat_winter = np.zeros(lat.shape)
    lat_summer = np.zeros(lat.shape)
    lon_winter = np.zeros(lon.shape)
    lon_summer = np.zeros(lon.shape)
    points_winter = np.zeros(points.shape)
    points_summer = np.zeros(points.shape)
    load_winter = np.zeros(load.shape)
    load_summer = np.zeros(load.shape)
    tide_winter = np.zeros(tide.shape)
    tide_summer = np.zeros(tide.shape)

    n = 0
    tmp = 15706
    for j in range(time.shape[1]):#pour chaque temps d'acquisition
        if time[0,j] <= tmp + pas_saison: #si le temps considéré est inférieur au jour de changement de saison (fixé comme jour repère + pas_saison)
            if n%2 == 0 : # si n pair => ete
                sla_summer[:,j] = sla[:,j]
                time_summer[:,j] = time[:,j]
                cycle_summer[j,:] = cycles[j,:]
                lat_summer[:] = lat[:]
                lon_summer[:] = lon[:]
                points_summer[:] = points[:]
                load_summer[:,j] = load[:,j]
                tide_summer[:,j] = tide[:,j]
                
            else: #si n impair => hiver
                sla_winter[:,j] = sla[:,j]
                time_winter[:,j] = time[:,j]
                cycle_winter[j,:] = cycles[j,:]
                lat_winter[:] = lat[:]
                lon_winter[:] = lon[:]
                points_winter[:] = points[:]
                load_winter[:,j] = load[:,j]
                tide_winter[:,j] = tide[:,j]
                
        else: #si le jour considéré est au delà de la date repère+pas_saison
            n+=1 #on passe à la saison suivante : si n pair, été (janvier-juin) / si n impair, hiver (juillet-décembre)
            tmp = tmp + pas_saison #mise a jour de la date repère
            if n%2 == 0 : # si n pair => ete
                sla_summer[:,j] = sla[:,j]
                time_summer[:,j] = time[:,j]
                cycle_summer[j,:] = cycles[j,:]
                lat_summer[:] = lat[:]
                lon_summer[:] = lon[:]
                points_summer[:] = points[:]
                load_summer[:,j] = load[:,j]
                tide_summer[:,j] = tide[:,j]
                
            else: #si n impair => hiver
                sla_winter[:,j] = sla[:,j]
                time_winter[:,j] = time[:,j]
                cycle_winter[j,:] = cycles[j,:]
                lat_winter[:] = lat[:]
                lon_winter[:] = lon[:]
                points_winter[:] = points[:]
                load_winter[:,j] = load[:,j]
                tide_winter[:,j] = tide[:,j]
                
    # Remplacer les 0 par des Nan 
    sla_winter = np.ma.masked_where(sla_winter==0., sla_winter)
    sla_summer = np.ma.masked_where(sla_summer==0., sla_summer)
    sla_winter = np.ma.masked_where(sla_winter>=100000, sla_winter)
    sla_summer = np.ma.masked_where(sla_summer>=100000, sla_summer)
    
    time_winter = np.ma.masked_where(time_winter==0., time_winter)
    time_summer = np.ma.masked_where(time_summer==0., time_summer)
    time_winter = np.ma.masked_where(time_winter>=10e+17, time_winter)
    time_summer = np.ma.masked_where(time_summer>=10e+17, time_summer)
    
    lat_winter = np.ma.masked_where(lat_winter==0., lat_winter)
    lat_summer = np.ma.masked_where(lat_summer==0., lat_summer)
    lat_winter = np.ma.masked_where(lat_winter==2147483647, lat_winter)
    lat_summer = np.ma.masked_where(lat_summer==2147483647, lat_summer)
    
    lon_winter = np.ma.masked_where(lon_winter==0., lon_winter)
    lon_summer = np.ma.masked_where(lon_summer==0., lon_summer)
    lon_winter = np.ma.masked_where(lon_winter==2147483647, lon_winter)
    lon_summer = np.ma.masked_where(lon_summer==2147483647, lon_summer)
    
    load_winter = np.ma.masked_where(load_winter==0., load_winter)
    load_summer = np.ma.masked_where(load_summer==0., load_summer)
    load_winter = np.ma.masked_where(load_winter>=10e+17, load_winter)
    load_summer = np.ma.masked_where(load_summer>=10e+17, load_summer)
    
    points_winter = np.ma.masked_where(points_winter==0., points_winter)
    points_summer = np.ma.masked_where(points_summer==0., points_summer)
    points_winter = np.ma.masked_where(points_winter==2147483647, points_winter)
    points_summer = np.ma.masked_where(points_summer==2147483647, points_summer)
    
    tide_winter = np.ma.masked_where(tide_winter==0., tide_winter)
    tide_summer = np.ma.masked_where(tide_summer==0., tide_summer)
    tide_winter = np.ma.masked_where(tide_winter>=10e+17, tide_winter)
    tide_summer = np.ma.masked_where(tide_summer>=10e+17, tide_summer)
    
    cycle_winter = np.ma.masked_where(cycle_winter==0., cycle_winter)
    cycle_summer = np.ma.masked_where(cycle_summer==0., cycle_summer)
    cycle_winter = np.ma.masked_where(cycle_winter==-1, cycle_winter)
    cycle_summer = np.ma.masked_where(cycle_summer==-1, cycle_summer)

    print(sla_winter)
    print(sla_summer)
    
    plt.figure()
    plt.plot(sla_winter[1000,:], label='winter')
    plt.plot(sla_summer[1000,:], label='summer')
    plt.legend()
    plt.show()

    #Sauvegarde des nouveaux fichiers
    winter = directory+'/winter/'+str(f[:-3])+"_winter.nc"
    output_w = nc.Dataset(winter,"w")
    output_w.createDimension('nbpoints', size=len(points))
    output_w.createDimension('nbcycles', size=cycles.shape[0])
    output_w.createDimension('Tracks', size=1)
    output_w.NCO = "\"4.5.4\""
    output_w.nco_openmp_thread_number = 1
    output_w.FileType = "ALONG_TRACK_PRODUCT"
    output_w.OriginalName = str(f[:-3])+"_winter.nc"
    output_w.title = "Residus TP"
    output_w.Mission = "TP"
    output_w.MeanProfile = "ProfilMoyen_TP_J1_J2_J3_2020.nc"

    cycles_w = output_w.createVariable('cycles','i4',('nbcycles','Tracks',), fill_value=-1)
    cycles_w.long_name = 'Cycle numbers for each pass'
    cycles_w.unit = "count"
    #cycles_w.set_auto_mask(False)
    cycles_w[:,:] = np.ma.filled(cycle_winter, -1)
    
    lat_w = output_w.createVariable('lat','i4',('nbpoints',),fill_value=2147483647)
    lat_w.long_name = 'Latitude of each measurement'
    lat_w.unit = "degrees_north"
    lat_w.scale_factor = 1.e-06
    #lat_w.set_auto_mask(False)
    lat_w[:] = np.ma.filled(lat_winter, 2147483647)
    
    lon_w = output_w.createVariable('lon','i4',('nbpoints',), fill_value=2147483647)
    lon_w.long_name = 'Longitude of each measurement'
    lon_w.units = "degrees_east"
    lon_w.scale_factor = 1.e-06
    #lon_w.set_auto_mask(False)
    lon_w[:] = np.ma.filled(lon_winter, 2147483647)
    
    load_w = output_w.createVariable('load', 'f4', ('nbpoints','nbcycles'), fill_value=1.844674e+19)
    load_w.long_name = 'Loading Tide FES2014b'
    load_w.units = "meters"
    #load_w.set_auto_mask(False)
    load_w[:,:] = np.ma.filled(load_winter, 1.844674e+19)
    
    point_w = output_w.createVariable('point','i4',('nbpoints',), fill_value=2147483647)
    point_w.long_name = 'Data index in theoretical pass'
    point_w.valid_min = 0
    #point_w.set_auto_mask(False)
    point_w[:] = np.ma.filled(points_winter, 2147483647)
    
    sla_w = output_w.createVariable('sla','f4',('nbpoints','nbcycles'), fill_value=1.844674e+19)
    sla_w.long_name = 'Sea Level Anomaly'
    sla_w.units = "meters"
    #sla_w.set_auto_mask(False)
    sla_w[:,:] = np.ma.filled(sla_winter, 1.844674e+19)
    
    tide_w = output_w.createVariable('tide','f4',('nbpoints','nbcycles'), fill_value=1.844674e+19)
    tide_w.long_name = 'Ocean Tide FES2014b'
    tide_w.units = "meters"
    #tide_w.set_auto_mask(False)
    tide_w[:,:] = np.ma.filled(tide_winter, 1.844674e+19)
    
    time_w = output_w.createVariable('time','f8', ('nbpoints', 'nbcycles'), fill_value=1.84467440737096e+19)
    time_w.units = "days since 1950-01-01 00:00:00.000 UTC"
    #time_w.set_auto_mask(False)
    time_w[:,:] = np.ma.filled(time_winter, 1.84467440737096e+19)
    
    output_w.set_auto_mask(False)
    output_w.close()

    summer = directory+'/summer/'+str(f[:-3])+"_summer.nc"
    output_s = nc.Dataset(summer,"w",format="NETCDF4", encoding='latin-1')
    output_s.createDimension('nbpoints', size=len(points))
    output_s.createDimension('nbcycles', size=cycles.shape[0])
    output_s.createDimension('Tracks', size=1)
    output_s.NCO = "\"4.5.4\""
    output_s.nco_openmp_thread_number = 1
    output_s.FileType = "ALONG_TRACK_PRODUCT"
    output_s.OriginalName = str(f[:-3])+"_winter.nc"
    output_s.title = "Residus TP"
    output_s.Mission = "TP"
    output_s.MeanProfile = "ProfilMoyen_TP_J1_J2_J3_2020.nc"
    
    cycles_s = output_s.createVariable('cycles','i4',('nbcycles','Tracks',), fill_value=-1)
    cycles_s.long_name = 'Cycle numbers for each pass'
    cycles_s.unit = "count"
    #cycles_s.set_auto_mask(False)
    cycles_s[:,:] = np.ma.filled(cycle_summer, -1)
    
    lat_s = output_s.createVariable('lat','i4',('nbpoints',), fill_value=2147483647)
    lat_s.long_name = 'Latitude of each measurement'
    lat_s.unit = "degrees_north"
    lat_s.scale_factor = 1.e-06
    #lat_s.set_auto_mask(False)
    lat_s[:] = np.m.filled(lat_summer, 2147483647)
    
    lon_s = output_s.createVariable('lon','i4',('nbpoints',),fill_value=2147483647)
    lon_s.long_name = 'Longitude of each measurement'
    lon_s.units = "degrees_east"
    lon_s.scale_factor = 1.e-06
    #lon_s.set_auto_mask(False)
    lon_s[:] = np.ma.filled(lon_summer, 2147483647)
    
    load_s = output_s.createVariable('load', 'f4',('nbpoints','nbcycles'), fill_value=1.844674e+19)
    load_s.long_name = 'Loading Tide FES2014b'
    load_s.units = "meters"
    #load_s.set_auto_mask(False)
    load_s[:,:] = np.ma.filled(load_summer,1.844674e+19)
    
    point_s = output_s.createVariable('point','i4',('nbpoints',), fill_value=2147483647)
    point_s.long_name = 'Data index in theoretical pass'
    point_s.valid_min = 0
    #point_s.set_auto_mask(False)
    point_s[:] = np.ma.filled(points_summer,2147483647)
    
    sla_s = output_s.createVariable('sla','f4',('nbpoints','nbcycles'), fill_value=1.844674e+19)
    sla_s.long_name = 'Sea Level Anomaly'
    sla_s.units = "meters"
    #sla_s.set_auto_mask(False)
    sla_s[:,:] = np.ma.filled(sla_summer,1.844674e+19)
    
    tide_s = output_s.createVariable('tide','f4',('nbpoints','nbcycles'), fill_value=1.844674e+19)
    tide_s.long_name = 'Ocean Tide FES2014b'
    tide_s.units = "meters"
    #tide_s.set_auto_mask(False)
    tide_s[:,:] = np.ma.filled(tide_summer,1.844674e+19)
    
    time_s = output_s.createVariable('time', 'f8',('nbpoints', 'nbcycles'), fill_value=1.84467440737096e+19)
    time_s.units = "days since 1950-01-01 00:00:00.000 UTC"
    #time_s.set_auto_mask(False)
    time_s[:,:] = np.ma.filled(time_summer, 1.84467440737096e+19)
    
    output_s.set_auto_mask(False)
    output_s.close()
    
data = nc.Dataset(directory+'/summer/Residus_t137_summer.nc')

sla = data.variables['sla'][:,:]
sla.mask=np.ma.nomask
print(sla)
plt.figure()
plt.plot(sla[1000,:])

plt.show()
