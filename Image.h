
#ifndef IMAGE_H_
#define IMAGE_H_

#include <opencv2/opencv.hpp>

#include "OCR.h"

using namespace std;
using namespace cv;

const int case_size = matching_size * 1.6;

const int crop_size = matching_size * 0.3;

int displayImage(const String display_name, Mat img);
Mat processInput(Mat img);
void transformScrabble(Mat img_src, Mat& img_tr, Rect& big_rect);
void segmentSudoku(Mat img, Mat& img_num, int x, int y);
void drawSudoku(Mat img, Rect rect, int grid[9][9]);


#endif /* IMAGE_H_ */
