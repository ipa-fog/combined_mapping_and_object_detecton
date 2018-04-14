/* *****************************************************************
 *
 * segmentation
 *
 * Copyright (c) %YEAR%,
 * Institute of Mechatronic Systems,
 * Leibniz Universitaet Hannover.
 * (BSD License)
 * All rights reserved.
 *
 * http://www.imes.uni-hannover.de
 *
 * This software is distributed WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.
 *
 * For further information see http://www.linfo.org/bsdlicense.html
 *
 ******************************************************************/

/**
 * @file   %FILENAME%
 * @author %USER% (%$EMAIL%)
 * @date   %DATE%
 *
 * @brief  PCL Region Growing Package, initialized by Bounding Boxes from darknet_ros
 */

#ifndef SEGMENTATION_SEGMENTATION_NODE_H
#define SEGMENTATION_SEGMENTATION_NODE_H

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_srvs/Empty.h"
#include "std_msgs/Int8.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/distortion_models.h>
#include <boost/make_shared.hpp>


//######################################### INCLUDE IMAGE PROCESSING HEADERS #########################################################
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <ros/console.h>
#include <visualization_msgs/Marker.h>

#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>

//######################################### YOLO DATA #################################################################################
#include <yolo2/ImageDetections.h>


using namespace cv;


class SegmentationNode
{
public:
    SegmentationNode(ros::NodeHandle &node_handle);

private:
    // ros communication
    ros::NodeHandle *node_;
    ros::Subscriber sub_yolo_bb_;
    ros::Subscriber sub_camera_depth_;
    ros::Subscriber sub_camera_info_;

    ros::Publisher pub_segm_cloud_;
    ros::Publisher pub_points_;
    ros::Publisher pub_name_;
    ros::Publisher pub_lines_;

    std::vector<std::string> names_;


    // Parameters from darknet_ros Bounding Box
    int x1;
    int x2;
    int y1;
    int y2;
    float confidence;
    int objectclass;
    std::string objectname;

    std::vector<uchar> array;
    std::vector<uchar> array2;

    yolo2::ImageDetections arrayBB;
    sensor_msgs::CameraInfo camerainfo;
    image_geometry::PinholeCameraModel pinmodel;

    // Parameters  for yaml
    double factorBBSize_;
    int GaussianBlurSize_;
    int MaximumFilterSize_;

    // callbacks
    void BBoxesCallback(const yolo2::ImageDetections::ConstPtr& bb);
    void cloud_segment (const sensor_msgs::PointCloud2::ConstPtr& cloud_in);
    void camera_info (const sensor_msgs::CameraInfo::ConstPtr& info);
    void readFile();

};

#endif // SEGMENTATION_SEGMENTATION_NODE_H
