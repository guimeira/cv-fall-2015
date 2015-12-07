#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    char* imageName = argv[1];

    Mat imgOriginal;
    Mat imgOriginalUnscaled;
    imgOriginalUnscaled = imread( imageName, 1 );

    int rows_Num = imgOriginalUnscaled.rows;
    if(rows_Num > 700) {
        double ScaleFactor = 700.0/rows_Num;
        resize(imgOriginalUnscaled, imgOriginal, cvSize(0, 0), ScaleFactor, ScaleFactor);
    } else {
        imgOriginal = imgOriginalUnscaled;
    }


    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int iLowB = 0;
    int iHighB = 40;

    int iLowG = 0; 
    int iHighG = 40;

    int iLowR = 0;
    int iHighR = 40;

    int iLowI = 0;
    int iHighI = 60;

      //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowB, 255); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighB, 255);

    cvCreateTrackbar("LowS", "Control", &iLowG, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighG, 255);

    cvCreateTrackbar("LowV", "Control", &iLowR, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighR, 255);

    cvCreateTrackbar("LowI", "Control", &iLowI, 255); //Value (0 - 255)
    cvCreateTrackbar("HighI", "Control", &iHighI, 255);


    // Mat imgHSV;
    Mat imgGRAY;

    // cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    cvtColor(imgOriginal, imgGRAY, COLOR_BGR2GRAY); //Convert the captured frame from BGR to HSV

    Mat imgThresholded;
    Mat imgThresholdedG;

    inRange(imgOriginal, Scalar(iLowB, iLowG, iLowR), Scalar(iHighB, iHighG, iHighR), imgThresholded); //Threshold the image
    inRange(imgGRAY, Scalar(iLowI), Scalar(iHighI), imgThresholdedG); //Threshold the image

    int kernal_size_x = (imgGRAY.cols)/70;
    int kernal_size_y = (imgGRAY.rows)/70;

    int kernal_size_x_o = (imgGRAY.cols)/70;
    int kernal_size_y_o = (imgGRAY.rows)/70;

    //morphological opening (remove small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x_o, kernal_size_y_o)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x_o, kernal_size_y_o)) ); 

    //morphological closing (fill small holes in the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x, kernal_size_y)) ); 
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x, kernal_size_y)) );

    //morphological opening (remove small objects from the foreground)
    erode(imgThresholdedG, imgThresholdedG, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x, kernal_size_y_o)) );
    dilate( imgThresholdedG, imgThresholdedG, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x, kernal_size_y_o)) ); 

    //morphological closing (fill small holes in the foreground)
    dilate( imgThresholdedG, imgThresholdedG, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x, kernal_size_y)) ); 
    erode(imgThresholdedG, imgThresholdedG, getStructuringElement(MORPH_ELLIPSE, Size(kernal_size_x, kernal_size_y)) );


    imwrite(argv[2],imgThresholded);

    while (true)
    {
        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        imshow("Gray Thresholded Image", imgThresholdedG); //show the thresholded image
        imshow("Gray", imgGRAY); //show the original image
        imshow("Original", imgOriginal); //show the original image

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break; 
        }
    }

   return 0;

}