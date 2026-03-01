#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


struct cameraIntrinsics {
    double fx;
    double fy;
    double cx;
    double cy;
};

struct pose {
    cv::Mat R; // Rotation matrix (3x3)
    cv::Mat t; // Translation vector (3x1)
};

pose estimateRelativePose(cv::Mat& grayframe,
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2,
    const cameraIntrinsics& intrinsics
    
);

int rotationMatrixToAngle(const cv::Mat& R);

void visualizeInliersAndOutliers(cv::Mat& grayframe, const std::vector<cv::Point2f>& points1,
                                const std::vector<cv::Point2f>& points2,
                                const cv::Mat& inlierMask);

std::vector<cv::Point3d> triangulatePoints(const std::vector<cv::Point2f>& points1,
                                    const std::vector<cv::Point2f>& points2,
                                    const cameraIntrinsics& intrinsics,
                                    cv::Mat inlierMask,
                                    const cv::Mat& R, const cv::Mat& t);