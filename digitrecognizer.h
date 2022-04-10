//
// Created by Anshumali Karna on 10/04/22.
//
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/types_c.h>
using namespace cv;
using namespace cv::ml;
#define MAX_NUM_IMAGES 60000 \


class DigitRecognizer
{
public:
    DigitRecognizer();

    ~DigitRecognizer();

    bool train(char* trainPath, char* labelsPath);

    int classify(Mat img);

private:
    Mat preprocessImage(Mat img);

    int readFlippedInteger(FILE *fp);

public:
    KNearest *knn;
    int numRows, numCols, numImages;

};
