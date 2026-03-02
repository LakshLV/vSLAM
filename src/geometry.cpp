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
        result.inlierMask = cv::Mat::zeros(points1.size(), 1, CV_8U); // No inliers
        return result;
    }
    cv::Mat inlierMask;
    // Step 1: Compute Essential Matrix
    if(points1.size() != points2.size()) {
        std::cerr << "ERROR: points1.size() (" << points1.size() 
                  << ") != points2.size() (" << points2.size() << ")" << std::endl;
        pose result;
        result.R = cv::Mat::eye(3, 3, CV_64F);
        result.t = cv::Mat::zeros(3, 1, CV_64F);
        return result;  // return identity pose
    }
    cv::Mat E = cv::findEssentialMat(points1, points2, intrinsics.fx, cv::Point2d(intrinsics.cx, intrinsics.cy), cv::FM_RANSAC, 0.99, 1.0, inlierMask);
    for(int i = 0; i < inlierMask.rows; i++) {
        if (inlierMask.at<uchar>(i,0)) {
            std::cout << "Inlier Point Pair: (" << points1[i] << ", " << points2[i] << ")\n";
        }
    }

    
    // Step 2: Decompose Essential Matrix to get R and t
    cv::Mat R, t;
    cv::recoverPose(E, points1, points2, R, t, intrinsics.fx, cv::Point2d(intrinsics.cx, intrinsics.cy), inlierMask);
    pose result;
    result.R = R;
    result.t = t;
    result.inlierMask = inlierMask.clone();

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


std::vector<cv::Point3d> triangulatePoints(const std::vector<cv::Point2f>& points1,
                                    const std::vector<cv::Point2f>& points2,
                                    const cameraIntrinsics& intrinsics,
                                    cv::Mat inlierMask,
                                    const cv::Mat& R, const cv::Mat& t) {
    std::vector<cv::Point2f> inlierPoints1;
    std::vector<cv::Point2f> inlierPoints2;
    
    for(size_t i = 0; i < inlierMask.rows; i++){
        if(inlierMask.at<uchar>(i,0)){
            inlierPoints1.push_back(points1[i]);
            inlierPoints2.push_back(points2[i]);
        }
    }

    // create projection matrices
    cv::Mat P1 = (cv::Mat_<double>(3, 4) << intrinsics.fx, 0, intrinsics.cx, 0,
                                            0, intrinsics.fy, intrinsics.cy, 0,
                                            0, 0, 1, 0);
    cv::Mat R64, t64;
    if(R.type() == CV_32F) R.convertTo(R64, CV_64F);
    if(t.type() == CV_32F) t.convertTo(t64, CV_64F);
    
    P1.type() == CV_32F ? P1.convertTo(P1, CV_64F) : P1;
    

    cv::Mat k = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.cx,
                                    0, intrinsics.fy, intrinsics.cy,
                                    0, 0, 1);

    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P2 = k * Rt;

    P2.type() == CV_32F ? P2.convertTo(P2, CV_64F) : P2;

    //prepare points for triangulation by creating 2xN matrices
    cv::Mat points1Mat(2, inlierPoints1.size(), CV_64F);
    cv::Mat points2Mat(2, inlierPoints2.size(), CV_64F);

    points1Mat.type() == CV_32F ? points1Mat.convertTo(points1Mat, CV_64F) : points1Mat;
    points2Mat.type() == CV_32F ? points2Mat.convertTo(points2Mat, CV_64F) : points2Mat;

    std::cout << "P1 size: " << P1.rows << " x " << P1.cols << std::endl;
    std::cout << "P2 size: " << P2.rows << " x " << P2.cols << std::endl;
    std::cout << "points1Mat size: " << points1Mat.rows << " x " << points1Mat.cols << std::endl;
    std::cout << "points2Mat size: " << points2Mat.rows << " x " << points2Mat.cols << std::endl;
    for(size_t i = 0; i < inlierPoints1.size(); i++){
        points1Mat.at<double>(0,i) = inlierPoints1[i].x;
        points1Mat.at<double>(1,i) = inlierPoints1[i].y;
        points2Mat.at<double>(0,i) = inlierPoints2[i].x;
        points2Mat.at<double>(1,i) = inlierPoints2[i].y;
    }

    cv::Mat points4D;
    if(inlierPoints1.size() < 5 || inlierPoints2.size() < 5){
        std::cout << "Not enough inliers for triangulation. Returning empty point cloud." << std::endl;
        return std::vector<cv::Point3d>();
    }
    cv::triangulatePoints(P1, P2, points1Mat, points2Mat, points4D);
    std::vector<cv::Point3d> points3D;
    for(int i = 0; i < points4D.cols; i++){
        cv::Vec4d point = points4D.col(i);
        if(point[3] != 0){
            points3D.push_back(cv::Point3d(point[0]/point[3], point[1]/point[3], point[2]/point[3]));
            std::cout << "Triangulated point (homogeneous): " << point << std::endl;
            std::cout << "Triangulated z value: " << points3D.back().z << std::endl;
            if(points3D.back().z < 0){
                std::cout << "Triangulated point is behind the camera: " << points3D.back() << std::endl;
                points3D.pop_back(); // Remove the point that is behind the camera
            }
        }
        
    }

    return points3D;
                        
}
