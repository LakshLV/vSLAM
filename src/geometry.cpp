#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "geometry.h"
#include <opencv2/calib3d.hpp>


pose estimateRelativePose(
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cameraIntrinsics& intrinsics
){
    cv::Mat inlierMask;
    // Step 1: Compute Essential Matrix
    cv::Mat E = cv::findEssentialMat(points1, points2, intrinsics.fx, cv::Point2d(intrinsics.cx, intrinsics.cy), cv::FM_RANSAC, 0.99, 3.0, inlierMask);
    std::cout << "Essential Matrix:\n" << E << std::endl;
    for(int i = 0; i < inlierMask.rows; i++) {
        if (inlierMask.at<uchar>(i)) {
            std::cout << "Inlier Point Pair: (" << points1[i] << ", " << points2[i] << ")\n";
        }
    }


    // Step 2: Decompose Essential Matrix to get R and t
    cv::Mat R, t;
    cv::recoverPose(E, points1, points2, R, t, intrinsics.fx, cv::Point2d(intrinsics.cx, intrinsics.cy));
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
    return static_cast<int>(angle * 180.0 / CV_PI); // Convert to degrees

    
}
