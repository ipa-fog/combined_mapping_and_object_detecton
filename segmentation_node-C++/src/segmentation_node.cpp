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
 * @brief  Filedescription
 */

#include "segmentation/segmentation_node.h"
#include "ros/package.h"

//########## CONSTRUCTOR ###############################################################################################
SegmentationNode::SegmentationNode(ros::NodeHandle &node_handle):
    node_(&node_handle)
{
    // === PARAMETERS ===
    node_->param("segmentation/factorBBSize", factorBBSize_, 0.0);
    node_->param("segmentation/GaussianBlurSize", GaussianBlurSize_, 0);
    node_->param("segmentation/MaximumFilterSize", MaximumFilterSize_, 0);


    readFile();

    // === SUBSCRIBERS ===

    sub_yolo_bb_ = node_->subscribe("/vision/yolo2/detections", 1, &SegmentationNode::BBoxesCallback, this);
    sub_camera_depth_ = node_->subscribe("/camera/depth_registered/points", 1, &SegmentationNode::cloud_segment, this);
    sub_camera_info_ = node_->subscribe("/camera/rgb/camera_info", 1, &SegmentationNode::camera_info, this);


    // === PUBLISHERS ===
    pub_segm_cloud_ = node_->advertise<sensor_msgs::PointCloud2>("segmentation/depth_registered/points", 1);
    pub_points_ = node_->advertise<visualization_msgs::Marker>("/segmentation/visualization_marker", 1);
    pub_name_ = node_->advertise<visualization_msgs::Marker>("/segmentation/visualization_name", 1);
    pub_lines_ = node_->advertise<visualization_msgs::Marker>("/segmentation/visualization_lines", 1);


}
//########## READ OBJECT NAMES FROM FILE ######################################################################################
void SegmentationNode::readFile()
{
    std::string file = ros::package::getPath("segmentation");
    file.append("/data/coco.names");

    std::ifstream f;  // Datei-Handle
    std::string name;
    f.open(file.c_str(), std::ios::in); // Öffne Datei aus Parameter
    while (!f.eof())   // Solange noch Daten vorliegen
    {
        std::getline(f, name);        // Lese eine Zeile
        names_.push_back(name);  // Zeige sie auf dem Bildschirm
    }
    f.close();    // Datei wieder schließen
}

//########## CALLBACK: SUBSCRIBER ######################################################################################
void SegmentationNode::BBoxesCallback(const yolo2::ImageDetections::ConstPtr &bb)
{
     // get Bounding Box data from yolo
     arrayBB = *bb;
     bb->header.stamp;
}

void SegmentationNode::camera_info (const sensor_msgs::CameraInfo::ConstPtr&info)
{
    //get camera infos for pinhole model
    camerainfo = *info;
}



void SegmentationNode::cloud_segment (const sensor_msgs::PointCloud2::ConstPtr& cloud_in)
{
    if (!arrayBB.detections.empty())
    {
        //load and encoding Pointcloud from ROS message
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pcl (new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg (*cloud_in, *cloud_pcl);

        //Create empty matrix & pointer with size of z-channel
        cv::Mat depth_image(cloud_pcl->height, cloud_pcl->width, CV_32FC1);
        float* depth_ptr = depth_image.ptr<float>(0);

        //fill z-values from pointcloud into pointer
        float maxDistance = 3.0;
        for (uint i =0; i < cloud_pcl->points.size();++i)
        {
            depth_ptr[i] = cloud_pcl->points[i].z;

           /* //get max Distance [m] in z-direction
            if (cloud_pcl->points[i].z>maxDistance)
            {
               maxDistance=cloud_pcl->points[i].z;
            }
            */
        }

        //transform distance values [m] in gray scale 0-255
        cv::Mat depth_image_8bit;
        depth_image.convertTo(depth_image_8bit, CV_8U, 256/maxDistance);


        //loop for each detected Bounding Box
        for (uint i=0; i<arrayBB.detections.size();++i)
        {
            //get class name[i] and ID[i] from /vision/yolo2/detections
            objectclass=arrayBB.detections[i].class_id;
            objectname = names_[arrayBB.detections[i].class_id];

            //get condidence[i] from /vision/yolo2/detections
            confidence=arrayBB.detections[i].confidence;

            //Transform from BB middle point, width and height to BB min / max points *****include scale factor*******
            x1 = (uint)((arrayBB.detections[i].x - (0.5*factorBBSize_*arrayBB.detections[i].width))*depth_image_8bit.cols);
            x2 = (uint)((arrayBB.detections[i].x + (0.5*factorBBSize_*arrayBB.detections[i].width))*depth_image_8bit.cols);
            y1 = (uint)((arrayBB.detections[i].y - (0.5*factorBBSize_*arrayBB.detections[i].height))*depth_image_8bit.rows);
            y2 = (uint)((arrayBB.detections[i].y + (0.5*factorBBSize_*arrayBB.detections[i].height))*depth_image_8bit.rows);

            //BB must be inside Image Borders
            if(x1 < 0)
            {
                 x1= 0;
            }

            if(x2 > depth_image_8bit.cols)
            {
                 x2= depth_image_8bit.cols;
            }
            if(y1 < 0)
            {
                 y1= 0;
            }

            if(y2 > depth_image_8bit.rows)
            {
                  y2= depth_image_8bit.rows;
            }

            ROS_INFO(" objectname %s, %i %i %i %i", objectname.c_str(),x1,x2,y1,y2);

            // Select ROI in depth image
            Point BB_topleft (x1,y1), BB_downright (x2,y2);
            Mat roi(depth_image_8bit, Rect(BB_topleft,BB_downright));

            /* OpenCV calc-Hist:
             * void calcHist(const Mat* images, int nimages, const int* channels, InputArray mask, OutputArray hist, int dims, const int* histSize, const float** ranges, bool uniform=true, bool accumulate=false )
             */

            // parameter initialization
            int histSize = 256;
            float range[] = {2, 253} ;
            const float* histRange = {range};
            Mat depth_hist;

            calcHist( &roi, 1, 0, Mat(), depth_hist, 1, &histSize, &histRange, true, false );

            // prepare depth histogram for drawing
            int hist_w = 500; int hist_h = 500;
            int bin_w = cvRound( (double) hist_w/histSize );
            Mat histImage( hist_h, hist_w, CV_8UC1, 1 );
            normalize(depth_hist, depth_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

            //Filtering histogram with gaussianblur
            cv::GaussianBlur(depth_hist, depth_hist, cv::Size(1,GaussianBlurSize_), 1.0);// 1D case

            //Filtering histogram with maximum filter
            int winSize = MaximumFilterSize_;
            int winMidSize = winSize / 2;
            for(int i = winMidSize; i < histSize - winMidSize; ++i)
            {
                float m = 0;
                for(int j = i - winMidSize; j <= (i + winMidSize); ++j)
                {
                    if (depth_hist.at<float>(i) > m)
                    {
                        m = depth_hist.at<float>(i);
                    }
                }
                depth_hist.at<float>(i) = m;
            }

            // Draw histogram
            for( int i = 1; i < histSize; i++ )
            {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(depth_hist.at<float>(i-1)) ) ,
                Point( bin_w*(i), hist_h - cvRound(depth_hist.at<float>(i)) ),
                Scalar( 255), 2, 8, 0  );
            }
/*
        // show histogram
            namedWindow("calcHist", WINDOW_AUTOSIZE );
            imshow("calcHist Demo", histImage );
            waitKey(1);

*/
            //find z-Borders for segmentation
            float maxhist=0.0; //initialize max hist value
            int index_maxhist=0; //initialize index of max hist value
            for (int i=0; i<histSize;++i) //search global maximum in histogram
            {
                if ( depth_hist.at<float>(i) > maxhist)
                {
                    maxhist=depth_hist.at<float>(i);
                    index_maxhist=i;
                }
            }

            int hist_upper_border=index_maxhist, hist_lower_border=index_maxhist;
            bool border_low=false, border_upper=false;

            while(border_low!=true)
            {
                if(depth_hist.at<float>(hist_lower_border)<=2.0)
                {
                    border_low=true;
                }
                if(border_low==false)
                {
                    hist_lower_border = hist_lower_border -1;
                }
            }

            while(border_upper!=true)
            {
                if(depth_hist.at<float>(hist_upper_border)<=2.0)
                {
                    border_upper=true;
                }
                if(border_upper==false)
                {
                hist_upper_border = hist_upper_border +1;
                }
            }



            int valRoi =1;
            float z1, z2;
            z1 = hist_lower_border*maxDistance/256;
            z2 = hist_upper_border*maxDistance/256;
            ROS_INFO("border low: %i border upper %i",hist_lower_border, hist_upper_border);
            ROS_INFO("border low in m: %f border upper in m %f",z1,z2);

            if(hist_lower_border == 0)
            {
                z1= z2 =maxDistance;
            }

            for(int u=0; u<roi.rows;u++)
            {
                for(int v=0; v<roi.cols;v++)
                {
                    if(roi.at<uchar>(u,v) > (uchar)hist_lower_border && roi.at<uchar>(u,v) < (uchar)hist_upper_border)
                    {
                        roi.at<uchar>(u,v)=valRoi;
                    }
                 }
            }

            // genereal visualization settings for points and text
            visualization_msgs::Marker points ,text, line;
            points.header.frame_id = text.header.frame_id = line.header.frame_id = cloud_in->header.frame_id;
            points.header.stamp =  text.header.stamp = line.header.stamp = ros::Time::now();
            points.action = text.action =text.action = visualization_msgs::Marker::ADD;
            points.pose.orientation.w = line.pose.orientation.w = text.pose.orientation.w = 1.0;
            points.id = text.id = line.id = objectclass; //set point id


            points.type = visualization_msgs::Marker::POINTS;
            points.scale.x = 0.01; // set size as cube in x
            points.scale.y = 0.01; // set size as cube in y
            points.color.b = 1.0f; // set color
            points.color.r = objectclass/80; // set color
            points.color.g = objectclass/80;
            points.color.a = 0.6;  // set transparency


            int numberofpixel=0;

            //transform depth image to pointcloud
            array.assign(depth_image_8bit.datastart, depth_image_8bit.dataend);
            for (int k=0;k<cloud_pcl->points.size();++k)
            {
                //ROS_INFO("[%i]",array[k]);
                if(array[k]==valRoi)
                {
                    numberofpixel = numberofpixel +1;

                    //draw points as Marker
                    geometry_msgs::Point p;
                    p.x = cloud_pcl->points[k].x;
                    p.y = cloud_pcl->points[k].y;
                    p.z = cloud_pcl->points[k].z;
                    points.points.push_back(p);

                    if(objectclass==0)
                    {
                        cloud_pcl->points[k].x = 0;
                        cloud_pcl->points[k].y = 0;
                        cloud_pcl->points[k].z = 0;
                    }

                }

                else
                {
                   // cloud_pcl->points[k].x = 0;
                   // cloud_pcl->points[k].y = 0;
                   // cloud_pcl->points[k].z = 0;
                }
            }

            int a = 0.05*(x2-x1)*(y2-y1);
            if (numberofpixel<0.2*(x2-x1)*(y2-y1))
            {
                z1 = z2 = maxDistance;
            }

            ROS_INFO("confidence %f numberofpixel %i, %i",confidence,numberofpixel, a);




            if(z1==z2)
            {
                line.color.r =1.0; // set color

            }

            else
            {
                line.color.r =0.0f; // set color

            }

            line.type = visualization_msgs::Marker::LINE_STRIP;
            line.scale.x = 0.01;

            line.color.g =confidence; // set color
            line.color.b =1.0; // set color
            line.color.a = 1.0;  // set transparency

            // project points form BB in 3D via pinhole model
            pinmodel.fromCameraInfo(camerainfo);
            Point BB_topright (x2,y1), BB_downleft (x1,y2);

            // rect BB in z=1m
            cv::Point3d P_topleft = pinmodel.projectPixelTo3dRay(BB_topleft);
            cv::Point3d P_downright = pinmodel.projectPixelTo3dRay(BB_downright);
            cv::Point3d P_topright = pinmodel.projectPixelTo3dRay(BB_topright);
            cv::Point3d P_downleft = pinmodel.projectPixelTo3dRay(BB_downleft);


            // vertices front rect
            geometry_msgs::Point p1;
            p1.x= P_topleft.x*z1;
            p1.y= P_topleft.y*z1;
            p1.z= P_topleft.z*z1;

            geometry_msgs::Point p2;
            p2.x= P_topright.x*z1;
            p2.y= P_topright.y*z1;
            p2.z= P_topright.z*z1;

            geometry_msgs::Point p3;
            p3.x= P_downright.x*z1;
            p3.y= P_downright.y*z1;
            p3.z= P_downright.z*z1;

            geometry_msgs::Point p4;
            p4.x= P_downleft.x*z1;
            p4.y= P_downleft.y*z1;
            p4.z= P_downleft.z*z1;

            // vertices rear rect
            geometry_msgs::Point p5;
            p5.x= P_topleft.x*z2;
            p5.y= P_topleft.y*z2;
            p5.z= P_topleft.z*z2;

            geometry_msgs::Point p6;
            p6.x= P_topright.x*z2;
            p6.y= P_topright.y*z2;
            p6.z= P_topright.z*z2;

            geometry_msgs::Point p7;
            p7.x= P_downright.x*z2;
            p7.y= P_downright.y*z2;
            p7.z= P_downright.z*z2;

            geometry_msgs::Point p8;
            p8.x= P_downleft.x*z2;
            p8.y= P_downleft.y*z2;
            p8.z= P_downleft.z*z2;

            line.points.push_back(p1);
            line.points.push_back(p2);
            line.points.push_back(p3);
            line.points.push_back(p4);
            line.points.push_back(p1);
            line.points.push_back(p5);
            line.points.push_back(p6);
            line.points.push_back(p7);
            line.points.push_back(p8);
            line.points.push_back(p5);
            line.points.push_back(p8);
            line.points.push_back(p4);
            line.points.push_back(p3);
            line.points.push_back(p7);
            line.points.push_back(p6);
            line.points.push_back(p2);


            text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text.pose.position.x =0.5*(p1.x+p2.x);
            text.pose.position.y =p1.y - 0.2;
            text.pose.position.z =0.5*(z1+z2);
            text.pose.orientation.x = 0.0;
            text.pose.orientation.y = 0.0;
            text.pose.orientation.z = 0.0;
            text.pose.orientation.w = 1.0;
            std::stringstream ss;
            ss << objectname << " (";
            ss << (int)(confidence*100) << " %)";
            text.text=ss.str();
            text.scale.z = 0.1;
            text.color.r = 0.0f;
            text.color.g = 0.0f;
            text.color.b = 1.0f;
            text.color.a = 1.0f;
            text.lifetime = ros::Duration();


            //publish markers
            pub_points_.publish(points);
            pub_name_.publish(text);
            pub_lines_.publish(line);

        }

        //publish optional modified point cloud 2
        sensor_msgs::PointCloud2 cloud_out;
        pcl::toROSMsg(*cloud_pcl, cloud_out);
        cloud_out.header = cloud_in->header;

        //publish modified PointCloud2 as sensor msgs
        pub_segm_cloud_.publish (cloud_out);
    }

     else
    {
        //publish original PointCloud2 as sensor msgs
        pub_segm_cloud_.publish (cloud_in);

        ROS_INFO("no object detected");
    }
}

//########## MAIN ######################################################################################################
int main(int argc, char** argv)
{
    ros::init(argc, argv, "segmentation_node");

    ros::NodeHandle node_handle;
    SegmentationNode segmentation_node(node_handle);

    ROS_INFO("Node is spinning...");
    ros::spin();


    return 0;
}
