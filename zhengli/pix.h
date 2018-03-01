#ifndef PIX_H
#define PIX_H
#include<opencv2/core/core.hpp>
#include<opencv2/ximgproc.hpp>
#include<opencv2/ml.hpp>
#include<QString>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/objdetect.hpp>
#include<opencv2/tracking.hpp>
#include<QDebug>
#include "kcf/kcftracker.hpp"
#define COLUNMS 576
#define ROWS 4947
class pix
{
private:
    cv::Ptr<cv::ml::SVM> svm;
    std::vector<cv::Rect> rects;
    int number,area;
    cv::Mat koutu,koutu1,koutu2;
    cv::Mat mask,labels,predictdata;
    double pdata[COLUNMS];
    float trainingData[1][COLUNMS];
    cv::Mat src,ce,gray,frame;
    cv::Rect rect;
    int minlen;
    int label;
    float scale;
    int ii,i1,i2;
    cv::Ptr<cv::FeatureDetector> pointdector;
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contoursImage;
    int num,inner,innermax;
    cv::Ptr<cv::HOGDescriptor> hog;
    std::vector<float> descripter;
    double lanMin,lanMax;
    std::vector<cv::MatND> check;
    cv::MatND histND;
    bool isdetection;
    float thresim;
    std::vector<cv::MatND>::iterator itND;
    KCFTracker tracker;
public:

    pix(int training=1)
    {
        hog=new cv::HOGDescriptor(cv::Size(64,64),cv::Size(16,16),cv::Size(8,8),cv::Size(8,8),9);
        if(training!=1)
        {
            svm=cv::ml::SVM::create();
            svm = cv::ml::SVM::load("trainingresult.xml");
            qDebug()<<"load";
        }
        isdetection=true;
        thresim=0.5;
        num=-1;
    }
    void predict2(cv::Mat &frame,cv::Mat &m);
    void predictjubu(cv::Mat &frame);
    void findhogdescriptor2();
    bool checkispositive();
    void calcsparseHist();
    void detectandtracking(cv::Mat &frame, cv::Mat &m);

};

#endif // PIX_H
