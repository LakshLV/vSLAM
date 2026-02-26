#include <opencv2/opencv.hpp>
#include <iostream>
#include "features.h"

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


            // cv::calcOpticalFlowPyrLK(
            //     prevgray,
            //     grayframe,
            //     prevpoints,
            //     nextPoints,
            //     status,
            //     err
            // );

            // Draw tracks and features
            drawTracks(grayframe, features, prevpoints, nextPoints, status);
            drawFeatures(grayframe, features);

            // std::vector<Feature> newFeatures;
            // for (size_t i = 0; i < nextPoints.size(); i++) {
            //     if (status[i] && i < features.size()) {
            //         Feature f = features[i];
            //         f.point = nextPoints[i];
            //         f.age += 1;
            //         newFeatures.push_back(f);
            //     }
            // }
            
            // features = newFeatures;

            if(features.size() < (size_t)MAX_CORNERS / 2) {
                detectInitialFeatures(
                    grayframe, features, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE,
                    BLOCK_SIZE, USE_HARRIS_DETECTOR, HARRIS_K, nextId, mask);
            }

            cv::imshow("VSLAM", grayframe);

            

            

            // Compute cell distribution
            // computeCellCounts(features, cellCount, cellwidth, cellheight,
            //                 GRID_ROWS, GRID_COLS);

            // std::cout << "\n--- Cell Distribution ---" << std::endl;
            // for (int i = 0; i < GRID_ROWS; i++) {
            //     for (int j = 0; j < GRID_COLS; j++) {
            //         std::cout << "Cell[" << i << "][" << j << "]: "
            //                  << cellCount[i * GRID_COLS + j] << " ";
            //     }
            //     std::cout << std::endl;
            // }
            // mask = cv::Mat::ones(frame.size(), CV_8UC1) * 255; // Reset mask
            // // Add new features if needed to balance cell distribution
            // addFeaturesByCell(grayframe, mask, features, MAX_CORNERS,
            //                 QUALITY_LEVEL, MIN_DISTANCE, BLOCK_SIZE,
            //                 USE_HARRIS_DETECTOR, HARRIS_K, nextId,
            //                 cellCount, cellwidth, cellheight,
            //                 GRID_ROWS, GRID_COLS, MIN_FEATURES_PER_CELL);

            

            // // Remove features if any cell exceeds max features per cell
            // removeFeaturesByCell(features, cellwidth, cellheight,
            //                     GRID_ROWS, GRID_COLS, MAX_FEATURES_PER_CELL);

            //std::cout << "Total features: " << features.size() << "\n" << std::endl;

            

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