#include <opencv2/opencv.hpp>
#include <iostream>
#include "features.h"
#include "geometry.h"

int main() {
    // Configuration
    const int MAX_CORNERS = 100;
    const double QUALITY_LEVEL = 0.01;
    const double MIN_DISTANCE = 30;
    const int BLOCK_SIZE = 3;
    const bool USE_HARRIS_DETECTOR = false;
    const double HARRIS_K = 0.04;
    const int GRID_ROWS = 4;
    const int GRID_COLS = 4;
    const int MIN_FEATURES_PER_CELL = 5;
    const int MAX_FEATURES_PER_CELL = 10;
    const int IMAGE_SIZE = 2000;


    //Initialization
    cv::VideoCapture cap("C:/Users/Laksh/projects/VSLAM/data/video.mp4");

    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open video." << std::endl;
        return -1;
    }

    cv::Mat frame, grayframe, prevgray;
    cap >> frame;

    if (frame.empty()) {
        std::cerr << "ERROR: Could not read first frame." << std::endl;
        return -1;
    }

    cameraIntrinsics intrinsics;
            intrinsics.fx = frame.cols / 2.0;
            intrinsics.fy = frame.cols / 2.0;
            intrinsics.cx = frame.cols / 2.0;
            intrinsics.cy = frame.rows / 2.0;

    // World pose initialization
    cv::Mat T_world = cv::Mat::eye(4,4,CV_64F);
    std::vector<cv::Mat> trajectory;
    trajectory.push_back(T_world.clone());

    std::vector<cv::Point3d> globalMap;


    cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY);

    double cellwidth = frame.cols / (double)GRID_COLS;
    double cellheight = frame.rows / (double)GRID_ROWS;

    // Initialize mask (white = trackable area)
    cv::Mat mask = cv::Mat::ones(frame.size(), CV_8UC1) * 255;

    std::vector<Feature> features;
    std::vector<cv::Point2f> prevpoints;
    std::vector<int> cellCount;
    int nextId = 0;

    // Detect Initial Features
    detectInitialFeatures(
        grayframe, features, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE,
        BLOCK_SIZE, USE_HARRIS_DETECTOR, HARRIS_K, nextId, mask);

    drawFeatures(grayframe, features);
    cv::imshow("VSLAM", grayframe);

    prevgray = grayframe.clone();
    prevpoints = extractPoints(features);

    // Main Loop
    while (true) {
        int key = cv::waitKey(30);

        // Process next frame (press 'd')  
        if (key == 100) {  // 'd' key
            cap >> frame;
            if (frame.empty()) {
                std::cout << "End of video" << std::endl;
                break;
            }
             
            cv::cvtColor(frame, grayframe, cv::COLOR_BGR2GRAY);

            // Track features using optical flow
            std::vector<cv::Point2f> nextPoints;
            std::vector<uchar> status;
            std::vector<float> err;

            

            features = trackFeatures(prevgray, grayframe, prevpoints, features,
                                   nextPoints, status, err);


            if(features.size() < (size_t)MAX_CORNERS / 2) {
                detectInitialFeatures(
                    grayframe, features, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE,
                    BLOCK_SIZE, USE_HARRIS_DETECTOR, HARRIS_K, nextId, mask);
            }


            
            pose relativePose = estimateRelativePose(frame, prevpoints, nextPoints, intrinsics);

            cv::Mat T = cv::Mat::eye(4,4,CV_64F);
            relativePose.R.copyTo(T(cv::Rect(0,0,3,3)));
            relativePose.t.copyTo(T(cv::Rect(3,0,1,3)));
            
            T_world = T_world * T;

            trajectory.push_back(T_world.clone());

            //triangulate points and add to global map with outlier rejection
            std::vector<cv::Point3d> newMapPoints = triangulatePoints(prevpoints, nextPoints, intrinsics, relativePose.inlierMask, relativePose.R, relativePose.t);

            std::cout << "Triangulated 3D Points:\n";
            for (const auto& pt : newMapPoints) {
                std::cout << pt << std::endl;
            }
            for(auto& pt : newMapPoints) {

                cv::Mat pt_cam = (cv::Mat_<double>(4,1) << pt.x, pt.y, pt.z, 1.0);
                cv::Mat pt_world = T_world * pt_cam;

                globalMap.push_back(cv::Point3d(
                    pt_world.at<double>(0),
                    pt_world.at<double>(1),
                    pt_world.at<double>(2)
                ));
            }

            

            //rotationMatrixToAngle(relativePose.R);


            cv::imshow("VSLAM", frame);


            // Update for next iteration
            prevgray = grayframe.clone();
            prevpoints = extractPoints(features);
        }

        // Exit (press 'esc')
        if (key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}