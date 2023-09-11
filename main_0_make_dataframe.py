# import argparse
from datetime import datetime
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
# import plotly.express as px
import pandas as pd
import scipy.spatial.distance
# from scipy.stats import chisquare
# import shapely
# from shapely.geometry import Polygon
# from sympy.geometry import Point, Polygon
import tqdm
import utm
import warnings
from xml.dom import minidom

import utils

warnings.simplefilter('always')

output_dataframe_filename = 'dataframe_annotations__all.pkl'

image_folders = ['C:/Vejdirektorat/images_for_annotation/20200817_test_ved_Skanderborg__completed',
                 'C:/Vejdirektorat/images_for_annotation/2020-08-24__completed',
                 #'C:/Vejdirektorat/images_for_annotation/2020-09-14__completed',
                 'C:/Vejdirektorat/images_for_annotation/2020-09-15__completed',
                 'C:/Vejdirektorat/images_for_annotation/2020-10-06']


def _to_utm(latitude, longitude):
    east, north, zone, zone_letter = utm.from_latlon(latitude, longitude)
    east = np.longdouble(east)
    north = np.longdouble(north)
    north = north if (zone_letter >= 'N') else np.longdouble(-1.0)*north
    return east, north, zone

def assign_clusters(adjacency_matrix, assigned=[], queue=[]):
    if len(assigned) == 0:
        assigned = np.asarray([-1 for i in range(adjacency_matrix.shape[0])])
    while len(queue) == 0  and np.any(assigned == -1):
        queue = [np.where(assigned == -1)[0][0]]
        assigned, queue = assign_clusters(adjacency_matrix, assigned, queue)
        queue.pop()
    
    if len(queue) > 0:
        neighbours = np.where(adjacency_matrix[queue[-1],:] > 0)[0]
        # Check if any previous neighbours have been assigned clusters
        neighbour_clusters = [assigned[neighbour] for neighbour in neighbours if assigned[neighbour] > -1]
        neighbour_unique_clusters = np.unique(neighbour_clusters)
        if len(neighbour_unique_clusters) == 0:
            # No neighbours have been assigned a cluster yet
            this_cluster = np.max(assigned)+1
        elif len(neighbour_unique_clusters) == 1:
            # One neighbour have been assigned a cluster
            this_cluster = neighbour_unique_clusters[0]
        elif len(neighbour_unique_clusters) > 1:
            raise RuntimeError('More than 1 cluster have been assigned to neighbours. queue: ' + ','.join([str(q) for q in queue]) + '; neighbour_clusters: ' + ','.join([str(nc) for nc in neighbour_clusters]))
            # TODO: Merge clusters...

        assigned[queue[-1]] = this_cluster

        for neighbour in neighbours:
            if neighbour in queue:
                # Neighbour is already in queue. Skip for now.
                continue
            if assigned[neighbour] == -1:
                assigned[neighbour] = this_cluster
                queue.append(neighbour)
                assigned, queue = assign_clusters(adjacency_matrix, assigned, queue)
                queue.remove(neighbour)

    return assigned, queue

def PolyArea(x,y):
    # Shoelace formula
    # Source: https://stackoverflow.com/a/30408825
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def bbox_area(bbox):

    # If xi2 or yi2 are not larger than xi1 or yi1, the bboxes are not intersection
    return max(0, bbox[2] - bbox[0])*max(0, bbox[3] - bbox[1])

def bbox_intersection(bbox1, bbox2):
    # bbox1 = [left bottom top right]
    # bbox2 = [left bottom top right]

    # Find intersection bbox
    xi1 = max(bbox1[0], bbox2[0])
    yi1 = max(bbox1[1], bbox2[1])
    xi2 = min(bbox1[2], bbox2[2])
    yi2 = min(bbox1[3], bbox2[3])

    return [xi1, yi1, xi2, yi2]

def bboxs_IoU(bbox1, bbox2):
    # bbox1 = [left bottom top right]
    # bbox2 = [left bottom top right]

    # Find intersection bbox
    intersecting_bbox = bbox_intersection(bbox1, bbox2)

    # Find area of intersection
    intersection_area = bbox_area(intersecting_bbox)

    # Find area of bboxes
    bbox1_area = bbox_area(bbox1)
    bbox2_area = bbox_area(bbox2)

    return intersection_area / (bbox1_area + bbox2_area - intersection_area)

DFs = []

# Load annotations
for img_folder in image_folders:
    folder_name = os.path.split(img_folder)[1]
    # image_list = glob.glob(os.path.join(img_folder,'*.jpg'))
    annotations_xml = glob.glob(os.path.join(img_folder,'annotations*.xml'))
    if (len(annotations_xml) < 1):
        warnings.warn('Could not detect any annotation xml-files in folder (' + folder_name + ').', Category=RuntimeWarning)
    elif (len(annotations_xml) > 1):
        raise RuntimeError('Detected more than 1 annotation xml-file in folder (' + folder_name + '). Could not determine which one is correct.')

    annotations_xml = annotations_xml[0]

    annotations = minidom.parse(annotations_xml)
    annotation_images = annotations.getElementsByTagName('image')

    df_annotations = pd.DataFrame(columns=['image','folder','ComputerTime','GPStime','latitude','longitude','east','north', 'zone','width','height','label','labels','N_polygons','polygons','Polygon_areas','Polygon_area','BBox_IoA','BBox_IoA_max'])
    for image in tqdm.tqdm(annotation_images, desc='Parsing ' + os.path.split(annotations_xml)[-1]):
        long_name = image.attributes['name'].value
        folder, image_name = os.path.split(long_name)

        # if image_name == 'GT_2020-10-06T08_43_49.000Z_CT_1597333318.6732814_9.716108_55.937814667.jpg':
        #     print('HERE')
        #     # Polygon overlap!
        #     pass

        image_name_parts = image_name.split('_')

        GT = '-'.join(image_name_parts[1:4])[:-1] + '000' # Add zeros to convert miliseconds to microseconds for later parsing
        GT_datetime = datetime.strptime(GT, '%Y-%m-%dT%H-%M-%S.%f')
        CT = float(image_name_parts[5])
        longitude = float(image_name_parts[6])
        latitude = float('.'.join(image_name_parts[7].split('.')[:-1]))

        east, north, zone = _to_utm(latitude, longitude)

        width = np.int(image.attributes['width'].value)
        height = np.int(image.attributes['height'].value)
        polygon_DOMs = image.getElementsByTagName('polygon')
        N_polygons = len(polygon_DOMs)

        polygons = []
        labels = []
        polygon_areas = []
        bboxs = []
        for polygon_DOM in polygon_DOMs:
            points = np.asarray([[float(c) for c in p.split(',')] for p in polygon_DOM.attributes['points'].value.split(';')])
            bboxs.append([np.min(points[:,0]), np.min(points[:,1]), np.max(points[:,0]), np.max(points[:,1])])
            # points = [point for p,point in enumerate(points) if point not in points[:p]] # Remove dublicate points. Only keep first
            # polygon = Polygon(points) #.buffer(0)
            polygons.append(points)
            polygon_areas.append(PolyArea(points[:,0], points[:,1]))
            label = polygon_DOM.attributes['label'].value
            labels.append(label)

        bbox_IoA = np.zeros((N_polygons, N_polygons))
        for b1, bbox1 in enumerate(bboxs):
            bbox1_area = bbox_area(bbox1)
            for b2 in range(b1+1, N_polygons):
                bbox2 = bboxs[b2]
                bbox2_area = bbox_area(bbox2)
                bboxI = bbox_intersection(bbox1, bbox2)
                bboxI_area = bbox_area(bboxI)
                bbox_IoA[b1,b2] = bboxI_area/bbox1_area
                bbox_IoA[b2, b1] = bboxI_area/bbox2_area

        # # Calculate intersection over union between all polygons in image
        # IoUs = []
        # for p1, polygon in enumerate(polygons):
        #     for p2 in range(p1+1, N_polygons):
        #         pU = polygon.union(polygons[p2])
        #         pI = polygon.intersection(polygons[p2])
        #         IoUs.append(pI.area/pU.area)
        
        # # Calculate intersection area over polygon area for all combination of polygons in image
        # IoA = np.zeros((N_polygons, N_polygons))
        # for p1, polygon1 in enumerate(polygons):
        #     for p2, polygon2 in enumerate(polygons):
        #         pI = polygon1.intersection(polygon2)
        #         IoA[p1,p1] = pI.area/polygon1.area

        label = np.unique(labels)
        if (len(label) > 1):
            label = 'Mixed'
        elif (len(label) == 0):
            label = 'None'
        else:
            label = label[0]

        df_annotations = df_annotations.append({'image': image_name, 'folder': folder, 'Date': GT_datetime.date(), 'ComputerTime': CT, 'GPStime': GT_datetime, 'latitude': latitude,'longitude': longitude, 'east': east, 'north': north, 'zone': zone, 'width': width, 'height': height, 'label': label, 'labels': labels, 'N_polygons':N_polygons, 'polygons': polygons, 'Polygon_area': polygon_areas, 'Polygon_area': np.sum(polygon_areas), 'BBox_IoA': bbox_IoA, 'BBox_IoA_max': np.max(bbox_IoA, initial=0.0)}, ignore_index=True)
    
    DFs.append(df_annotations)

df_all = pd.concat(DFs, ignore_index=True)

# Calculate distances between all points
D = scipy.spatial.distance.pdist(np.asarray([df_all['east'], df_all['north']]).transpose(),'euclidean')
# Show histogram of "small" distances (< 40.0 m)
plt.hist(D, bins=41, range=(0.0, 40.0))
plt.show()

# # Convert distances to square matrix and plot
D2 = scipy.spatial.distance.squareform(D)
# plt.imshow(D2, vmin=0.0, vmax=40.0)
# plt.show()

clusters, queue = assign_clusters((D2 < 40)*D2)

clusters = clusters.astype(np.uint16)

df_all['cluster'] = clusters
df_all['dates'] = [d.__str__() for d in df_all['Date']]

df_all.to_pickle(output_dataframe_filename)

# df_all['Polygon_IoUs'].map(lambda d: np.sum(d))

utils.print_annotation_stats(df_all)

print('done')