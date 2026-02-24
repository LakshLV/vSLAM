#include <opencv2/opencv.hpp>
#include <iostream>

int main(){
    
    
    // Open video file
    cv::VideoCapture cap(
    "C:/Users/Laksh/projects/VSLAM/data/video.mp4"
);


    if (!cap.isOpened())
    {
        std::cerr << "ERROR: Could not open video." << std::endl;
        return -1;
    }

    cv::Mat frame;

    
    cap >> frame; // read first frame
    

    while(true) {

        int key = cv::waitKey(30);
        
        //cap >> frame; // read next frame

        if (frame.empty())
            break;

        cv::imshow("VSLAM Viewer", frame);

        if(key == 100){ // d key goes forward a frame
            cap >> frame;
        }

        // cap << frame; // go back a frame

        if(key == 97){ // a key goes back a frame
             cap.set(cv::CAP_PROP_POS_FRAMES, cap.get(cv::CAP_PROP_POS_FRAMES) - 2);
                cap >> frame;
        }
         
         if (key == 27) { // ESC key
             std::cout << "ESC key pressed. Exiting." << std::endl;
             break;
         } else {
             std::cout << "Key pressed: " << key << std::endl;
         }
    }
}