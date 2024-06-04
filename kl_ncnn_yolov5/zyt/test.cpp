//
// Created by zyt on 3/15/23.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Open the camera
    VideoCapture cap("http://172.23.208.107:1234/?action=stream");

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream" << endl;
        return -1;
    }

    // Loop through the video stream
    while (true) {
        // Read a new frame from the camera
        Mat frame;
        cap.read(frame);

        // Draw a rectangle on the frame
        rectangle(frame, Point(100, 100), Point(200, 200), Scalar(0, 0, 255), 2);

        // Convert the frame to JPEG format
        vector<uchar> jpeg;
        imencode(".jpg", frame, jpeg);

        // Send the JPEG frame to mjpg-streamer
        cout << "Content-Type: image/jpeg\r\n";
        cout << "Content-Length: " << jpeg.size() << "\r\n";
        cout << "\r\n";
        fwrite(jpeg.data(), 1, jpeg.size(), stdout);
        fflush(stdout);

        // Wait for a short time to limit the frame rate
        usleep(100000);
    }

    // Release the camera
    cap.release();

    return 0;
}
