#include "pix.h"
#include<iostream>
#include<opencv2\core\core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\ml\ml.hpp>
#include<opencv2/ximgproc.hpp>
#include<opencv2/video/tracking.hpp>
#include<strstream>
#include<QDateTime>
#include<QDebug>
#include<QFile>
#include<opencv2/xfeatures2d.hpp>
using namespace std;
using namespace cv;
void pix::predict2(Mat &frame, Mat &m)
{
    rects.clear();
    //cv::medianBlur(frame,frame,9);
    //frame.copyTo(this->frame);
    cv::cvtColor(frame,src,CV_BGR2GRAY);
    //m=cv::Mat(frame.rows,frame.cols,CV_8UC1,cv::Scalar(0));
    src.copyTo(gray,cv::Mat());
    for(ii=0;ii<gray.cols;ii++)
    {
       for(i1=0;i1<gray.rows;i1++)
       {
           double p= (float)gray.at<uchar>(i1,ii);
           p = pow((p)/255.0,0.5f)*255;//
           gray.at<uchar>(i1,ii) = p;
       }
    }

    cv::Laplacian(gray,koutu1,CV_32F,7);//最后一个参数为核矩阵的大小7为7x7
    cv::minMaxLoc(koutu1,&lanMin,&lanMax);
    //缩放保证不平坦为黑，平坦处为128
    scale=-127/std::max(-lanMin,lanMax);
    koutu1.convertTo(koutu1,CV_8U,scale,128);
    cv::threshold(koutu1,koutu1,120,255,cv::THRESH_BINARY_INV);
    //cv::dilate(lanplace,lanplace,cv::Mat(),cv::Point(-1,-1),1);
    //cv::erode(lanplace,lanplace,cv::Mat(),cv::Point(-1,-1),1);
    cv::threshold(gray,koutu2,200,255,CV_THRESH_BINARY);

    cv::erode(koutu2,koutu2,cv::Mat(),cv::Point(-1,-1),1);
    cv::dilate(koutu2,koutu2,cv::Mat(),cv::Point(-1,-1),1);
    koutu=koutu1+koutu2;
    cv::dilate(koutu,koutu,cv::Mat(),cv::Point(-1,-1),3);
    cv::erode(koutu,koutu,cv::Mat(),cv::Point(-1,-1),3);
    ce=cv::Mat(gray.rows,gray.cols,CV_8UC1,cv::Scalar(255));
    src.copyTo(ce,koutu);
    cv::threshold(ce,m,100,255,CV_THRESH_BINARY_INV);
    cv::erode(m,m,cv::Mat(),cv::Point(-1,-1),1);
    cv::dilate(m,m,cv::Mat(),cv::Point(-1,-1),10);
    findContours(m,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    for(ii=0;ii<contours.size();ii++)
    {
        rect=cv::boundingRect(cv::Mat(contours[ii]));
        for(int i=0;i<2;i++)
        {
            scale=0.1*i+1.1;
            if(rect.height<rect.width)
            {

                minlen=rect.height*scale;
                int x=rect.x-(minlen-rect.height)/2.0;
                int y=rect.y-(minlen-rect.height)-minlen/10.0;
                if(minlen>5&&minlen+x<gray.cols&&minlen+y<gray.rows&&x>0&&y>0)
                {
                    /*while(rect.width<minlen)
                    rect.width++;
                while(rect.height<minlen)
                    rect.height++;*/
                    rect.width=minlen;
                    rect.height=minlen;
                    rect.x=x;
                    rect.y=y;
                }
                findhogdescriptor2();
                label=svm->predict(predictdata);
                if(label==1)
                {

                    rects.push_back(rect);

                    break;
                }
                else
                {
//                    cv::imshow("meiy",frame(rect));
//                    cv::waitKey(100);
                }

            }
            else
            {
                minlen=rect.width*scale;
                int x=rect.x-(minlen-rect.width)/2.0;
                int y=rect.y-(minlen-rect.width)-minlen/10.0;

                if(minlen>5&&minlen+x<gray.cols&&minlen+y<gray.rows&&x>0&&y>0)
                {

                    rect.width=minlen;
                    rect.height=minlen;
                    rect.x=x;
                    rect.y=y;
                }
                findhogdescriptor2();
                label=svm->predict(predictdata);
                if(label==1)
                {

                    rects.push_back(rect);

                    break;
                }
                else
                {
//                    cv::imshow("meiy",frame(rect));
//                    cv::waitKey(100);
                }

            }
        }

    }

    num=-1;
    area=100;
    for(int i=0;i<rects.size();i++)
    {
        rect=rects.at(i);
        if(rect.area()>area)
        {
            area=rect.area();
            num=i;
        }
    }
    if(num!=-1)
    {

        rect=rects.at(num);
        //cv::rectangle(frame,rect,cv::Scalar(255),3);
    }
}

void pix::findhogdescriptor2()
{
    ce=src(rect);
    //cv::imshow("rect",ce);
    cv::resize(ce,ce,cv::Size(64,64));
    hog->compute(ce,descripter);
    for(i1=0;i1<COLUNMS;i1++)
    {
        trainingData[0][i1]=descripter[i1];

    }
    predictdata=cv::Mat(1,COLUNMS , CV_32FC1,trainingData);//可能需要存留数组
}

void pix::calcsparseHist()
{
    ce=frame(rect);
    int h_bins=50,s_bins=60;
    int histSize[]={h_bins,s_bins,s_bins};
    float h_ranges[]={0,180};
    float s_ranges[]={0,256};
    float v_ranges[]={0,256};
    const float * ranges[]={h_ranges,s_ranges,v_ranges};
    int channels[]={0,1,2};
    cv::cvtColor(ce,ce,CV_BGR2HSV);
    cv::calcHist(&ce,1,channels,cv::Mat(),histND,2,histSize,ranges);
}

bool pix::checkispositive()
{

    if(rects.size()!=0)
    {
        calcsparseHist();
        if(check.size()<10)
        {
            check.push_back(histND.clone());
            isdetection=true;

            return true;
        }
        else
        {
            double simliar=0;
            itND=check.begin();
            for(ii=0;itND!=check.end();ii++,itND++)
            {

                simliar=simliar+cv::compareHist(histND,*itND,0)*(0.05+0.01*ii);
            }

            if(!isdetection)
                qDebug()<<"simliar "<<simliar;
            if(simliar>thresim)
            {
                //qDebug()<<"66";
                check.erase(check.begin());
                check.push_back(histND.clone());
                if(isdetection)
                {
                    //tracker=cv::TrackerKCF::createTracker();;
                    tracker.init(rect,frame);
                }
                isdetection=false;
                if(thresim<0.9)
                thresim=simliar*0.3+thresim*0.7;
                qDebug()<<"thresim "<<thresim;
                return true;
            }
            else
            {
                if(!isdetection)
                {
//                    qDebug()<<"detection "<<simliar<<"   thresim "<<thresim;
                    thresim=0.5;
//                    if(thresim<0.4)
//                    {
//                        check.clear();
//                        thresim=0.5;
//                    }
                }
                else
                {
                    qDebug()<<"detection "<<simliar<<"   thresim "<<thresim;
                    if(thresim>0.4)
                    thresim=simliar*0.5+thresim*0.5;
//                    if(thresim<0.45)
//                        thresim=0.45;
                    if(thresim<0.4)
                    {
                        check.clear();
                        thresim=0.5;
                    }
//                    QString s="E:/QT/project/eigen/hardexample/"+QDateTime::currentDateTime().toString("mm_ss_z_")+QString::number(ii)+".jpg";
//                    cv::imwrite(s.toStdString(),gray(rect));
                }
                isdetection=true;
                qDebug()<<"thresim "<<thresim;
                //check.clear();
                return false;
            }
        }
    }
    return false;
}
void pix::detectandtracking(Mat &frame, Mat &m)
{
    frame.copyTo(this->frame);
    if(isdetection)
    {
        predict2(frame, m);
    }
    else
    {

        rects.clear();
        //cv::Rect2d rect2d;
        //rect2d=rect;
//        cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER,100,0.01);
//        cv::meanShift(gray,rect,criteria);
        rect=tracker.update(frame);

//        rect.x=rect2d.x;
//        rect.y=rect2d.y;
//        rect.width=rect2d.width;
//        rect.height=rect2d.height;
        if(rect.width+rect.x>frame.cols)
            rect.width=frame.cols-rect.x;
        if(rect.width<0)
            rect.width=0;
        if(rect.height+rect.y>frame.rows)
            rect.height=frame.rows-rect.y;
        if(rect.height<0)
            rect.height=0;
        if(rect.x<0)
            rect.x=0;
        if(rect.y<0)
            rect.y=0;
        cv::rectangle(frame,rect,cv::Scalar::all(0),3);
//        cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER,100,0.01);
//        cv::meanShift(gray,rect,criteria);
        rects.push_back(rect);
    }
    if(checkispositive())
    {
        cv::rectangle(frame,rect,cv::Scalar(255),3);
    }
}
void pix::predictjubu(cv::Mat &frame)
{
    cv::Mat m;
    if(num!=-1)
    {
        if(rect.x-rect.width<0)
            rect.x=0;
        else
            rect.x=rect.x-rect.width;
        if(rect.y-rect.height<0)
            rect.y=0;
        else
            rect.y=rect.y-rect.height;
        if(rect.x+rect.width*3>=frame.cols)
            rect.width=frame.cols-rect.x;
        else
            rect.width=rect.width*3;
        if(rect.y+rect.height*3>=frame.rows)
            rect.height=frame.rows-rect.y;
        else
            rect.height=rect.height*3;
        cv::Mat f=frame(rect);
        int x,y;
        x=rect.x;
        y=rect.y;
        predict2(f,m);
        if(num!=-1)
        {
            rect.x=rect.x+x;
            rect.y=y+rect.y;
            cv::rectangle(frame,rect,cv::Scalar(255),3);
        }
    }
    else
    {
        predict2(frame,m);
        if(num!=-1)
        {
            cv::rectangle(frame,rect,cv::Scalar(255),3);
        }
    }
}
