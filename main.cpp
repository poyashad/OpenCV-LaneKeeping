#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
Mat roi(Mat src, Rect rect){
    Mat roi;
    rectangle(src, rect, Scalar::all(255), 2); // bara för att se vilken yta jag jobbar med
    roi = src(rect);
    return roi;
}
Mat convertToBGR(Mat processedFrame) {
    Mat frameInBGR;
    cvtColor(processedFrame, frameInBGR , COLOR_GRAY2BGR);
    return frameInBGR;
};

Mat processFrame(Mat frame){
    Mat grayscale, gaussianBlur, imageCanny, processedFrame;
    cvtColor(frame, grayscale, COLOR_BGR2GRAY);
    Canny(grayscale, imageCanny, 100, 300);
    GaussianBlur(imageCanny, gaussianBlur, Size(5, 5), 0);
    vector<Vec4i> lines;
    HoughLinesP(gaussianBlur, lines, 1, CV_PI/180, 50, 10, 0);
    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        line( gaussianBlur, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,255), 1, LINE_AA);
    }
    processedFrame = gaussianBlur;
    return processedFrame;
}

int main(int argc, char **argv) {

    string filename = "/Users/pojashad/Desktop/dev/opencvtest/motorcycle.mp4";
    VideoCapture cap(filename);
    //VideoCapture cap(1); // open the video camera no. 0
    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
    namedWindow("Processed", WINDOW_AUTOSIZE);
    while (1) {

        Mat src, processedFrame, ROI, processedFrameWithLines, frameBGR;
        bool bSuccess = cap.read(src); // read a new frame from video
        if (!bSuccess) {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        
        // Ta ut region of interest
        Rect rect(600, 400, 600, 400);
        ROI = roi(src, rect);
        // Process ROI
       processedFrame = processFrame(ROI);
        // Convertera tillbaka processerad ROI med linjer till BGR för kunna lägga till i src frame:n
        frameBGR = convertToBGR(processedFrame);
        // Merge src filen och den processerade ROI
        frameBGR.copyTo(src(rect));
        imshow("Processed", src);
        if (waitKey(30) == 27) {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }
    return 0;
}
