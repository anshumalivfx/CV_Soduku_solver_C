//
// Created by Anshumali Karna on 10/04/22.
//
#include "digitrecognizer.h"
typedef unsigned char BYTE;
using namespace std;

DigitRecognizer::DigitRecognizer()
{
    knn = cv::ml::KNearest::create();
}

DigitRecognizer::~DigitRecognizer()
{
    delete knn;
}

int DigitRecognizer::readFlippedInteger(FILE *fp) {
    int ret = 0;

    BYTE *temp;

    temp = (BYTE*)(&ret);
    fread(&temp[3], sizeof(BYTE), 1, fp);
    fread(&temp[2], sizeof(BYTE), 1, fp);
    fread(&temp[1], sizeof(BYTE), 1, fp);

    fread(&temp[0], sizeof(BYTE), 1, fp);

    return ret;

}

bool DigitRecognizer::train(char *trainPath, char *labelsPath) {
    FILE *fp = fopen(trainPath, "r");
    FILE *fp2 = fopen(labelsPath, "r");
    if(!fp || !fp2) {
        return false;
    }

    int magicNumber = readFlippedInteger(fp);
    numImages = readFlippedInteger(fp);
    numRows = readFlippedInteger(fp);
    numCols = readFlippedInteger(fp);
    fseek(fp, 0x08, SEEK_SET);
    if(numImages > MAX_NUM_IMAGES) numImages = MAX_NUM_IMAGES;

    int size = numRows*numCols;


    memset(trainingVectors, 0, sizeof(Mat)*numImages);

    BYTE *temp = new BYTE[size];
    BYTE tempClass=0;

    for(int i=0;i<numImages;i++)
    {

        fread((void*)temp, size, 1, fp);

        fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

        trainingClasses->data.fl[i] = tempClass;

        for(int k=0;k<size;k++)
            trainingVectors->data.fl[i*size+k] = temp[k]; ///sumofsquares;

    }
    knn->train(trainingVectors, cv::ml::ROW_SAMPLE, trainingClasses);
    fclose(fp);

    fclose(fp2);

    return true;
}