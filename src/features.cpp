#include "features.h"
#include <iostream>
#include <algorithm>
#include <map>

// ============== Feature Creation ==============
Feature createFeature(cv::Point2f point, int id, int age) {
    Feature feature;
    feature.point = point;
    feature.id = id;
    feature.age = age;
    return feature;
}

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
    const cv::Mat& mask) {
    
    std::vector<cv::Point2f> detectedPoints;
    cv::goodFeaturesToTrack(
        grayframe,
        detectedPoints,
        maxCorners,
        qualityLevel,
        minDistance,
        mask,
        blockSize,
        useHarrisDetector,
        k
    );

    std::cout << "Detected " << detectedPoints.size() << " initial features" << std::endl;
    
    for (const auto& point : detectedPoints) {
        features.push_back(createFeature(point, nextId++, 0));
    }
}

// ============== Feature Tracking ==============
std::vector<Feature> trackFeatures(
    const cv::Mat& prevgray, const cv::Mat& grayframe,
    std::vector<cv::Point2f>& prevpoints,
    const std::vector<Feature>& features,
    std::vector<cv::Point2f>& nextPoints,
    std::vector<uchar>& status,
    std::vector<float>& err) {

    if (prevpoints.size() != features.size()) {
        std::cerr << "ERROR: prevpoints.size() (" << prevpoints.size() 
                  << ") != features.size() (" << features.size() << ")" << std::endl;
        return features;  // return unchanged
    }

    cv::calcOpticalFlowPyrLK(prevgray, grayframe, prevpoints, nextPoints, status, err);

    
    std::vector<Feature> trackedFeatures;
    std::vector<cv::Point2f> validNextPoints;
    std::vector<cv::Point2f> validPrevPoints;

    // Only keep features that were successfully tracked
    for(size_t i = 0; i < status.size() && i < nextPoints.size(); i++) {
        if (status[i] && i < features.size()) {
            Feature trackedFeature = features[i];
            trackedFeature.point = nextPoints[i];
            trackedFeature.age += 1; // Increment age for successfully tracked features
            trackedFeatures.push_back(trackedFeature);
            validNextPoints.push_back(nextPoints[i]);
            validPrevPoints.push_back(prevpoints[i]);
            std::cout << "tracked successfully feature id: " << trackedFeature.id << std::endl;
        }
    }

    prevpoints = validPrevPoints;
    nextPoints = validNextPoints;

    // std::cout << "Tracked " << trackedFeatures.size() << " out of " 
    //           << features.size() << " features" << std::endl;

    return trackedFeatures;
}

// Feature Visualization 
void drawFeatures(
    cv::Mat& grayframe,
    const std::vector<Feature>& features,
    const cv::Scalar& color,
    int radius,
    int thickness) {
    
    for (const auto& feature : features) {
        cv::circle(grayframe, feature.point, radius, color, thickness);
    }
}

void drawTracks(
    cv::Mat& grayframe,
    const std::vector<Feature>& features,
    const std::vector<cv::Point2f>& prevPoints,
    const std::vector<cv::Point2f>& nextPoints,
    const std::vector<uchar>& status) {

    
    for (size_t i = 0; i < status.size() && i < nextPoints.size(); i++) {
        if (status[i]) {
            // Draw line from previous to current position
            cv::line(grayframe, prevPoints[i], nextPoints[i],
                    cv::Scalar(0, 255, 0), 2);
            
            // Draw circle at current position
            cv::circle(grayframe, nextPoints[i], 3, cv::Scalar(0, 0, 255), -1);
            
            // Draw feature age
            cv::putText(grayframe,
                       std::to_string(features[i].age),
                       nextPoints[i],
                       cv::FONT_HERSHEY_SIMPLEX,
                       0.5,
                       cv::Scalar(255, 255, 255),
                       1);
            std::cout << "feature number: " << features[i].id << std::endl;
        }
    }
}

// ============== Cell-based Feature Management ==============
// void computeCellCounts(
//     const std::vector<Feature>& features,
//     std::vector<int>& cellCount,
//     double cellwidth,
//     double cellheight,
//     int rows,
//     int cols) {
    
//     cellCount.assign(rows * cols, 0);
    
//     for (const auto& feature : features) {
//         int cellX = std::min(std::max((int)(feature.point.x / cellwidth), 0), cols - 1);
//         int cellY = std::min(std::max((int)(feature.point.y / cellheight), 0), rows - 1);
//         int cellIndex = cellY * cols + cellX;
        
//         if (cellIndex >= 0 && cellIndex < (int)cellCount.size()) {
//             cellCount[cellIndex]++;
//         }
//     }
// }

// //brief - Add new features to underpopulated cells by:
// // 1. Extracting features inside a cell
// // 2. if the count is less than minFeaturesPerCell, detect new features in that cell and add them to the list
// // 3. if the count is greater than maxFeaturesPerCell, call removeFeaturesByCell to remove excess features from that specific cell
// void addFeaturesByCell(
//     const cv::Mat& grayframe,
//     const cv::Mat& mask,
//     std::vector<Feature>& features,
//     int maxCorners,
//     double qualityLevel,
//     double minDistance,
//     int blockSize,
//     bool useHarrisDetector,
//     double k,
//     int& nextId,
//     const std::vector<int>& cellCount,
//     int cellWidth,
//     int cellHeight,
//     int rows,
//     int cols,
//     int minFeaturesPerCell) {

//     int maxFeaturesPerCell = 10;
    
//     for (int cellY = 0; cellY < rows; cellY++) {
//         for (int cellX = 0; cellX < cols; cellX++) {
//             int cellIndex = cellY * cols + cellX;
//             if (cellIndex >= 0 && cellIndex < (int)cellCount.size()) {
//                 if (cellCount[cellIndex] < minFeaturesPerCell) { // Underpopulated cell, add new features
//                     // Define the region of interest for this cell
//                     cv::Rect roi(cellX * cellWidth, cellY * cellHeight, cellWidth, cellHeight);
//                     cv::Mat cellMask = mask(roi);
//                     cv::Mat cellGray = grayframe(roi);

//                     // Detect new features in this cell
//                     std::vector<cv::Point2f> newPoints;
//                     cv::goodFeaturesToTrack(
//                         cellGray,
//                         newPoints,
//                         maxCorners - cellCount[cellIndex],
//                         qualityLevel,
//                         minDistance,
//                         cellMask,
//                         blockSize,
//                         useHarrisDetector,
//                         k
//                     );

//                     // Add new features to the list
//                     for (const auto& point : newPoints) {
//                         cv::Point2f globalPoint = point + cv::Point2f(cellX * cellWidth, cellY * cellHeight);
//                         features.push_back(createFeature(globalPoint, nextId++, 0));
//                     }
//                 } else if (cellCount[cellIndex] > maxFeaturesPerCell) {
//                     // Remove excess features from this specific cell
//                     removeFeaturesByCell(features, cellWidth, cellHeight, rows, cols, maxFeaturesPerCell);
//                 }
//             }
//         }
//     }
// }

// void removeFeaturesByCell(
//     std::vector<Feature>& features,
//     double cellWidth,
//     double cellHeight,
//     int rows,
//     int cols,
//     int maxFeaturesPerCell) {
//     std::vector<std::vector<Feature>> cellFeatures(rows * cols);
//     for (const auto& feature : features) {
//         int cellX = std::min(std::max((int)(feature.point.x / cellWidth), 0), cols - 1);
//         int cellY = std::min(std::max((int)(feature.point.y / cellHeight), 0), rows - 1);
//         int cellIndex = cellY * cols + cellX;
//         if (cellIndex >= 0 && cellIndex < (int)cellFeatures.size()) {
//             cellFeatures[cellIndex].push_back(feature);
//         }
//     }

//     // Remove excess features from each cell
//     std::vector<Feature> filteredFeatures;
//     for (const auto& cell : cellFeatures) {
//         if (cell.size() > (size_t)maxFeaturesPerCell) { // If there are more features than the max allowed, we need to remove some
//             // Sort by age (ascending)
//             std::vector<Feature> sortedCell = cell;
//             std::sort(sortedCell.begin(), sortedCell.end(),
//                       [](const Feature& a, const Feature& b) {
//                           return a.age < b.age;
//                       });
//             // Keep only the maxFeaturesPerCell oldest features
//             int start = std::max(0, (int)sortedCell.size() - maxFeaturesPerCell);
//             for (size_t i = start; i < sortedCell.size(); i++) {
//                 filteredFeatures.push_back(sortedCell[i]); // Add excess features to the filtered list (these will be removed)
//             }
//         } else {
//             filteredFeatures.insert(filteredFeatures.end(), cell.begin(), cell.end());
//         }

//     }
//     features = filteredFeatures;
// }

// ============== Utility Functions ==============
std::vector<cv::Point2f> extractPoints(const std::vector<Feature>& features) {
    std::vector<cv::Point2f> points;
    for (const auto& feature : features) {
        points.push_back(feature.point);
    }
    return points;
}

void updateFeaturePoints(
    std::vector<Feature>& features,
    const std::vector<cv::Point2f>& newPoints) {
    
    if (features.size() != newPoints.size()) {
        std::cerr << "Warning: Feature count mismatch in updateFeaturePoints" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < features.size(); i++) {
        features[i].point = newPoints[i];
    }
}

