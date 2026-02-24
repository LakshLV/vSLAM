#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct Feature {
    cv::Point2f point;
    int id;
    int age;
};

// ============== Feature Creation ==============
Feature createFeature(cv::Point2f point, int id, int age);

// ============== Feature Detection ==============
void detectInitialFeatures(
    const cv::Mat& grayframe,
    std::vector<Feature>& features,
    int maxCorners,
    double qualityLevel,
    double minDistance,
    int blockSize,
    bool useHarrisDetector,
    double k,
    int& nextId,
    const cv::Mat& mask = cv::Mat()
);

// ============== Feature Tracking ==============
std::vector<Feature> trackFeatures(
    const cv::Mat& prevgray,
    const cv::Mat& grayframe,
    std::vector<cv::Point2f>& prevpoints,
    const std::vector<Feature>& features,
    std::vector<cv::Point2f>& nextPoints,
    std::vector<uchar>& status,
    std::vector<float>& err
);

// ============== Feature Visualization ==============
void drawFeatures(
    cv::Mat& grayframe,
    const std::vector<Feature>& features,
    const cv::Scalar& color = cv::Scalar(0, 0, 255),
    int radius = 5,
    int thickness = 2
);

void drawTracks(
    cv::Mat& grayframe,
    const std::vector<Feature>& features,
    const std::vector<cv::Point2f>& prevPoints,
    const std::vector<cv::Point2f>& nextPoints,
    const std::vector<uchar>& status
);

// ============== Cell-based Feature Management ==============
void computeCellCounts(
    const std::vector<Feature>& features,
    std::vector<int>& cellCount,
    double cellwidth,
    double cellheight,
    int rows,
    int cols
);

void addFeaturesByCell(
    const cv::Mat& grayframe,
    const cv::Mat& mask,
    std::vector<Feature>& features,
    int maxCorners,
    double qualityLevel,
    double minDistance,
    int blockSize,
    bool useHarrisDetector,
    double k,
    int& nextId,
    const std::vector<int>& cellCount,
    int cellWidth,
    int cellHeight,
    int rows,
    int cols,
    int minFeaturesPerCell = 5
);

void removeFeaturesByCell(
    std::vector<Feature>& features,
    double cellwidth,
    double cellheight,
    int rows,
    int cols,
    int maxFeaturesPerCell = 10
);  

// ============== Utility Functions ==============
std::vector<cv::Point2f> extractPoints(const std::vector<Feature>& features);

void updateFeaturePoints(
    std::vector<Feature>& features,
    const std::vector<cv::Point2f>& newPoints
);
