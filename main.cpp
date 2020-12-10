#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect.hpp>


using namespace cv;
using namespace std;

vector<Point2f> slidingWindow(Mat image, Rect window)
{
    vector<Point2f> points;
    const Size imgSize = image.size();
    bool shouldBreak = false;

    while (true)
    {
        float currentX = window.x + window.width * 0.5f;

        Mat roi = image(window); //Extract region of interest
        vector<Point2f> locations;

        findNonZero(roi, locations); //Get all non-black pixels. All are white in our case

        float avgX = 0.0f;

        for (int i = 0; i < locations.size(); ++i) //Calculate average X position
        {
            float x = locations[i].x;
            avgX += window.x + x;
        }

        avgX = locations.empty() ? currentX : avgX / locations.size();

        Point point(avgX, window.y + window.height * 0.5f);
        points.push_back(point);

        //Move the window up
        window.y -= window.height;

        //For the uppermost position
        if (window.y < 0)
        {
            window.y = 0;
            shouldBreak = true;
        }

        //Move x position
        window.x += (point.x - currentX);

        //Make sure the window doesn't overflow, we get an error if we try to get data outside the matrix
        if (window.x < 0)
            window.x = 0;
        if (window.x + window.width >= imgSize.width)
            window.x = imgSize.width - window.width - 1;

        if (shouldBreak)
            break;
    }
    return points;
}

int main(int argc, char **argv) {
    const String cascade_name("/Users/pojashad/Desktop/dev/opencvtest/cascade2.xml");
    CascadeClassifier cascade;
    if (!cascade.load(cascade_name)) {
        cout << "Error loading face cascade\n";
        return -1;
    }
    //---------------GET IMAGE---------------------
    // Read the image file
    // TODO: Change to an argument, enter path to image
    string filename = "/Users/pojashad/Desktop/dev/opencvtest/malibu.mp4";
    VideoCapture cap(filename);
    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the video cam" << endl;
        return -1;
    }
    // ------------------ Birds Eye View of the lane---------------------
    Point2f srcVertices[4];
    srcVertices[0] = Point(450, 480);
    srcVertices[1] = Point(730, 480);
    srcVertices[2] = Point(1000, 700);
    srcVertices[3] = Point(200, 700);

    //Destination vertices. Output is 640 by 480px
    Point2f dstVertices[4];
    dstVertices[0] = Point(0, 0);
    dstVertices[1] = Point(640, 0);
    dstVertices[2] = Point(640, 480);
    dstVertices[3] = Point(0, 480);

    //Prepare matrix for transform and get the warped image
    Mat perspectiveMatrix = getPerspectiveTransform(srcVertices, dstVertices);
    Mat dst(480, 640, CV_8UC3); //Destination for warped image

    //For transforming back into original image space
    Mat invertedPerspectiveMatrix;
    invert(perspectiveMatrix, invertedPerspectiveMatrix);


    Mat org; //Original image, modified only with result
    Mat img; //Working image


    int frameCounter = 0;
    int tick = 0;
    int fps;
    std::time_t timeBegin = std::time(0);

    while (1) {
        // Just to have a matrix of the original video
        Mat src;
        cap.read(src);

        //Read a frame
        cap.read(org);


        if (org.empty()) //When this happens we've reached the end
            break;

        //--------------Bird's eye view-----------------
        //Generate bird's eye view
        warpPerspective(org, dst, perspectiveMatrix, dst.size(), INTER_LINEAR, BORDER_CONSTANT);

        //--------------GRAYSCALE IMAGE-----------------
        // Define grayscale image
        //Convert to gray
        cvtColor(dst, img, COLOR_RGB2GRAY);

        // --------------Detect Yellow & White Lines-----------------
        //Extract yellow and white info
        Mat maskYellow, maskWhite;

        inRange(img, Scalar(20, 100, 100), Scalar(30, 255, 255), maskYellow);
        inRange(img, Scalar(150, 150, 150), Scalar(255, 255, 255), maskWhite);

        Mat mask, processed;
        bitwise_or(maskYellow, maskWhite, mask); //Combine the two masks
        bitwise_and(img, mask, processed); //Extrect what matches

        // --------------Gaussian Blur-----------------
        //Blur the image a bit so that gaps are smoother
        const Size kernelSize = Size(9, 9);
        GaussianBlur(processed, processed, kernelSize, 0);

        //Try to fill the gaps
        Mat kernel = Mat::ones(15, 15, CV_8U);
        dilate(processed, processed, kernel);
        erode(processed, processed, kernel);
        morphologyEx(processed, processed, MORPH_CLOSE, kernel);

        //Keep only what's above 150 value, other is then black
        const int thresholdVal = 175;
        threshold(processed, processed, thresholdVal, 255, THRESH_BINARY);
        //Might be optimized with adaptive thresh

        //Get points for left sliding window. Optimize by using a histogram for the starting X value
        vector<Point2f> pts = slidingWindow(processed, Rect(0, 420, 120, 60));
        vector<Point> allPts; //Used for the end polygon at the end.

        vector<Point2f> outPts;
        perspectiveTransform(pts, outPts, invertedPerspectiveMatrix); //Transform points back into original image space

        //Draw the points onto the out image
        for (int i = 0; i < outPts.size() - 1; ++i)
        {
            line(org, outPts[i], outPts[i + 1], Scalar(255, 0, 0), 3);
            allPts.emplace_back(outPts[i].x, outPts[i].y);
        }

        allPts.emplace_back(outPts[outPts.size() - 1].x, outPts[outPts.size() - 1].y);

        Mat out;
        cvtColor(processed, out, COLOR_GRAY2BGR); //Conver the processing image to color so that we can visualise the lines

        for (int i = 0; i < pts.size() - 1; ++i) //Draw a line on the processed image
            line(out, pts[i], pts[i + 1], Scalar(255, 0, 0));

        //Sliding window for the right side
        pts = slidingWindow(processed, Rect(520, 420, 100, 40));
        perspectiveTransform(pts, outPts, invertedPerspectiveMatrix);

        //Draw the other lane and append points
        for (int i = 0; i < outPts.size() - 1; ++i)
        {
            line(org, outPts[i], outPts[i + 1], Scalar(0, 0, 255), 3);
            allPts.emplace_back(outPts[outPts.size() - i - 1].x, outPts[outPts.size() - i - 1].y);
        }

        allPts.emplace_back(outPts[0].x - (outPts.size() - 1) , outPts[0].y);

        for (int i = 0; i < pts.size() - 1; ++i)
            line(out, pts[i], pts[i + 1], Scalar(0, 0, 255));

        //Create a green-ish overlay
        vector<vector<Point>> arr;
        arr.push_back(allPts);
        Mat overlay = Mat::zeros(org.size(), org.type());
        fillPoly(overlay, arr, Scalar(0, 255, 100));
        addWeighted(org, 1, overlay, 0.5, 0, org); //Overlay it

        //-------------- Object detection -----------------
        Mat frame_gray;
        cvtColor(org, frame_gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        cascade.detectMultiScale(frame_gray,faces);
        for (auto& face : faces) {
            rectangle(org, face, Scalar(255, 255, 0), 2);
            cv::putText(org, cv::format("Car"), cv::Point(face.x, face.y-10),
                        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0,0,255));
        }

        // Add frame counter to the org
        frameCounter++;
        std::time_t timeNow = std::time(0) - timeBegin;
        if (timeNow - tick >= 1)
        {
            tick++;
            fps = frameCounter;
            frameCounter = 0;
        }
        cv::putText(org, cv::format("Average FPS=%d", fps ), cv::Point(40, 40),
                    cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0,0,255));

        //Show results
        imshow("Preprocess", out);
        imshow("src", org);

        if (waitKey(30) == 27) {
            cout << "esc key is pressed by user" << endl;
            break;
        }

    }
    cap.release();
    return 0;
}
