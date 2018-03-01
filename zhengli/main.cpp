#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/ml.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/objdetect.hpp>
#include<opencv2/ximgproc.hpp>
#include<opencv2/video.hpp>
#include<QFile>
#include<QTextStream>
#include<strstream>
#include<QDebug>
#include"pix.h"
int labels[ROWS];
float trainingData[ROWS][COLUNMS];
void training();
void predict();
int main()
{
    predict();
    return 0;
}
void predict()
{
    std::strstream ss;
    bool fromFile=true;
    cv::VideoCapture capture;
    //cv::VideoWriter writer("VideoTest7.avi",CV_FOURCC('M','P','4','2'),25.0, cv::Size(640,480));
    if(!fromFile)
        capture.open(1);
    capture.set(CV_CAP_PROP_FOCUS, 1);
    std::string name;
    cv::namedWindow("predict");
    cv::Mat frame,mask,copy;
    mask=cv::Mat();
    frame = cv::imread(name);
   pix p(0);
    int i =0;
    int max;
    max=100000;
    while (i<=max)
    {
        if(fromFile)
        {
            ss.clear();
            name.clear();
            ss << i;
            ss >> name;

            switch (name.length())
            {
            case 1:
                name = "G:/ceshiji/soccer000" + name +".jpg";
                //name = "E:/QT/project/socc/soccer/soccer000" + name + ".jpg";
                break;
            case 2:
                name = "G:/ceshiji/soccer00" + name + ".jpg";
                //name = "E:/QT/project/socc/soccer/soccer00" + name + ".jpg";
                break;
            case 3:
                //name = "E:/QT/project/socc/soccer/soccer0" + name + ".jpg";
                name = "G:/ceshiji/soccer0" + name + ".jpg";
                break;
            case 4:
                //name = "E:/QT/project/socc/soccer/soccer" + name + ".jpg";
                name = "G:/ceshiji/soccer" + name + ".jpg";
                break;
            }
            frame = cv::imread(name);
        }
        else
        {
            capture.read(frame);
        }
        p.predictjubu(frame);
        //writer<<frame;
        cv::imshow("frame",frame);
        if(cvWaitKey(10)!=-1)
        {
            //cv::imwrite("xx.png",frame);
            return ;
        }
        cv::waitKey(1);
        i++;
    }

}

void training()
{
    QString file="E:/QT/project/eigen/training.txt";
    QFile f(file);
    f.open(QIODevice::ReadOnly);
    QTextStream ts(&f);
    QString str,stri,strj;
    str=ts.readAll();
    QStringList strlist,list,listi;
    strlist=str.split("\r\n",QString::SkipEmptyParts);//将字符串以“_”为界分割成数组形式存到QStringList变量strlist中
    for(int i=0;i<ROWS;i++)
    {
        str=strlist.at(i);
        list=str.split("_");
        stri=list.at(1);
        labels[i]=stri.toInt();
        stri=list.at(2);
        listi=stri.split(" ");
        qDebug()<<i<<"  "<<labels[i];
        for(int j=0;j<COLUNMS;j++)
        {
            strj=listi.at(j);
            trainingData[i][j]=strj.toFloat();
        }
    }
    int width = 512;
    int height = 512;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    // training data
    cv::Mat trainingDataMat(ROWS,COLUNMS, CV_32FC1, trainingData);
    cv::Mat labelsMat(ROWS, 1, CV_32SC1, labels);
    // initial SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(100);
    svm->setKernel(2);//0-l 2-k
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-10));

    // train operation
    //svm->setC(100);
    cv::Ptr<cv::ml::TrainData> data=cv::ml::TrainData::create(trainingDataMat,0,labelsMat);
    svm->trainAuto(data);
    labelsMat=cv::Mat();
    svm->predict(trainingDataMat,labelsMat);
    for(int i=0;i<labelsMat.rows;i++)
    {
        if(labelsMat.at<float>(i,0)==1)
        qDebug()<<i<<" "<<labelsMat.at<float>(i,0);
    }
    qDebug()<<svm->isClassifier();
    svm->save("trainingresult.xml");
    qDebug()<<"over";
    f.close();
}
