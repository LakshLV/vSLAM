#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "geometry.h"
#include <opencv2/calib3d.hpp>


pose estimateRelativePose(cv::Mat& grayframe,
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cameraIntrinsics& intrinsics
){

    //calculate pixel distance, if too small, skip pose estimation (this is a simple heuristic to avoid estimating pose when the camera is stationary or nearly stationary)
    double totalPixelDistance = 0.0;
    for (size_t i = 0; i < points1.size() && i < points2.size(); i++) {
        totalPixelDistance += cv::norm(points1[i] - points2[i]);
    }
    if (totalPixelDistance < 10.0) {
        std::cout << "Too small pixel distance: " << totalPixelDistance << std::endl;
        pose result;
        result.R = cv::Mat::eye(3, 3, CV_64F);
        result.t = cv::Mat::zeros(3, 1, CV_64F);
        return result;
    }
    cv::Mat inlierMask;
    // Step 1: Compute Essential Matrix
    cv::Mat E = cv::findEssentialMat(points1, points2, intrinsics.fx, cv::Point2d(intrinsics.cx, intrinsics.cy), cv::FM_RANSAC, 0.99, 1.0, inlierMask);
    std::cout << "Essential Matrix:\n" << E << std::endl;
    for(int i = 0; i < inlierMask.rows; i++) {
        if (inlierMask.at<uchar>(i,0)) {
            std::cout << "Inlier Point Pair: (" << points1[i] << ", " << points2[i] << ")\n";
        }
    }

    visualizeInliersAndOutliers(grayframe, points1, points2, inlierMask);

    // Step 2: Decompose Essential Matrix to get R and t
    cv::Mat R, t;
    cv::recoverPose(E, points1, points2, R, t, intrinsics.fx, cv::Point2d(intrinsics.cx, intrinsics.cy), inlierMask);
    std::cout << "Recovered Rotation Matrix:\n" << R << std::endl;
    std::cout << "Recovered Translation Vector:\n" << t << std::endl;
    pose result;
    result.R = R;
    result.t = t;

    return result;



}

int rotationMatrixToAngle(const cv::Mat& R) {
    // Assuming R is a valid rotation matrix
    double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
    double angle = std::acos((trace - 1) / 2);
    std::cout << "Rotation Matrix:\n" << R << std::endl;
    return (angle * 180.0 / CV_PI); // Convert to degrees

    
}

//Visualize inliers and outliers
void visualizeInliersAndOutliers(cv::Mat& grayframe, const std::vector<cv::Point2f>& points1,
                                const std::vector<cv::Point2f>& points2,
                                const cv::Mat& inlierMask) {
    
    for (int i = 0; i < inlierMask.rows; i++) {
        if (inlierMask.at<uchar>(i,0)) {
            std::cout << "Inlier: (" << points1[i] << ", " << points2[i] << ")\n";
            cv::line(grayframe, points1[i], points2[i], cv::Scalar(0, 255, 0), 2); // Green for inliers
            cv::circle(grayframe, points1[i], 3, cv::Scalar(0, 255, 0), -1); // Green circle for inliers
            cv::circle(grayframe, points2[i], 3, cv::Scalar(0, 255, 0), -1); // Green circle for inliers
        } else {
            std::cout << "Outlier: (" << points1[i] << ", " << points2[i] << ")\n";
            cv::line(grayframe, points1[i], points2[i], cv::Scalar(0, 0, 255), 2); // Red for outliers
            cv::circle(grayframe, points1[i], 3, cv::Scalar(0, 0, 255), -1); // Red circle for outliers
            cv::circle(grayframe, points2[i], 3, cv::Scalar(0, 0, 255), -1); // Red circle for outliers
        }
    }
}
