#include "opencv2/highgui.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace optflow;
using namespace std;

int main(int argc, char **argv)
{
    cout << "usage : inputdirA framenameA extensionA inputdirB framenameB extensionB loop_start loop_end loop_half_window" << endl;
    cout << "> ./looping ../images lionwalk png 94 120 4" << endl;
    cout << endl;
    
    char *inputdirA=             argv[1];
    char *framenameA=            argv[2];
    char *extensionA=            argv[3];
    char *inputdirB=             argv[4];
    char *framenameB=            argv[5];
    char *extensionB=            argv[6];
    int loop_start =            atoi(argv[7]);
    int loop_end =              atoi(argv[8]);
    int loop_half_window =      atoi(argv[9]);
    
    //numbers
    int loop_center            =loop_start+round((loop_end-loop_start)/2);
    int loop_offset            =loop_end-loop_start;
    int transition_start       =loop_center-loop_half_window;
    int transition_end         =loop_center+loop_half_window;
    
    cout << "center : " << loop_center << endl;
    cout << endl;
    
    //variables declaration
    char Aimage[1024];
    char Bimage[1024];
    char ASimage[1024];
    char BSimage[1024];
    char outputdir[1024];
    char outputimage[1024];
    char ABflowfile[1024];
    char BAflowfile[1024];
    char linuxcmd[1024];
    Mat imageA,imageB,simageA,simageB,bwA,bwB,mapx,mapy,warpA,warpB,finalframe;
    Mat_<Point2f> BAflow,ABflow;
    Ptr<DenseOpticalFlow> algorithm;
    double startTick,time;
    double averagetime=0;
    int timecount=0;
    
    //static flags
    int startatframe1=1; //output sequence starts at 1
    int useGpu = 1; //opencv deepflow uses gpu
    cv::ocl::setUseOpenCL(useGpu);
    int writeflo=1;
    //path to deepmatching and deepflow2 executables.adapt to your system.
    char deepmatching[] ="/shared/foss-18/looping/deepmatching-static";
    char deepflow[]     ="/shared/foss-18/looping/deepflow2-static";
    
    string input;
    int flowmethod=1;
    float flowscale =0.5;
    //downscale images for optical flow (.5 = half)
    cout << "downscale image for flow calculation[" << flowscale << "] : ";
    getline(cin,input);
    if ( !input.empty() ) {
        istringstream stream(input);
        stream >> flowscale;
    }
    //create outputdir
    sprintf(outputdir,"./looping_reverse_%s_%d_%d",framename,loop_start,loop_end);
    sprintf(linuxcmd,"mkdir %s",outputdir);
    cout << linuxcmd << endl;
    system(linuxcmd);
    
    //no flow , just copy
    for (int i = loop_start; i <= transition_start; i++) {
        int Aframe=i;
        sprintf(Aimage,"%s/%s.%04d.%s",inputdir,framename,Aframe,extension);
        cout << "processing loop frame : " << i-loop_start+1 << endl;
        cout <<"reading : " << Aimage << endl;
        imageA= imread(Aimage,IMREAD_COLOR);
        //writing result
        if (startatframe1 == 1) {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i-loop_start+1,extension);
        } else {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i,extension);
        }
        cout << "writing final frame : " << outputimage << endl;
        cv::imwrite(outputimage,imageA);
        cout << endl;
    }
    
    //using flow
    for (int i = transition_start + 1; i < transition_end; i++) {
        int Aframe=i;
        int Bframe=(2*loop_start)-i+loop_offset;
        sprintf(Aimage,"%s/%s.%04d.%s",inputdir,framename,Aframe,extension);
        sprintf(Bimage,"%s/%s.%04d.%s",inputdir,framename,Bframe,extension);
        cout << "processing loop frame : " << i-loop_start+1 << endl;
        //read A/B image
        cout <<"readingA : " << Aimage << endl;
        imageA= imread(Aimage,IMREAD_COLOR);
        cout <<"readingB : " << Bimage << endl;
        imageB= imread(Bimage,IMREAD_COLOR);
        //compute fractional value
        float frac=(float((i-transition_start)/float((transition_end-transition_start))));
        cout << "fractional : " << frac << endl;
        
        if (flowmethod != 0) {
            startTick = (double) getTickCount(); // measure time
            if (flowmethod == 1) { //use opencv deep optical flow
                algorithm = createOptFlow_DeepFlow();
                cout << "using opencv deepflow algorithm" << endl;
                cvtColor(imageA, bwA, COLOR_BGR2GRAY);
                cvtColor(imageB, bwB, COLOR_BGR2GRAY);
                //resize for optical flow computation
                if (flowscale != 1) {
                    cv::resize(bwA,bwA,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cv::resize(bwB,bwB,cv::Size(),flowscale,flowscale,INTER_AREA);
                    cout <<"flow scale : " << flowscale << " [" << bwA.size[1] << "," << bwA.size[0] << "]" << endl;
                }
                BAflow = Mat(bwA.size[0], bwA.size[1], CV_32FC2);
                ABflow = Mat(bwB.size[0], bwB.size[1], CV_32FC2);
                //process optical flow
                cout << "processing B->A flow  : " << Bframe << "->" << Aframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwB, bwA, BAflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwB, bwA, BAflow);
                cout << "processing A->B flow  : " << Aframe << "->" << Bframe << " [useGpu : " << int(useGpu && cv::ocl::haveOpenCL()) << "]" << endl;
                if (useGpu) algorithm->calc(bwA, bwB, ABflow.getUMat(ACCESS_RW));
                else algorithm->calc(bwA, bwB, ABflow);
                //resize flow for warping
                if (flowscale != 1) {
                    cv::resize(BAflow,BAflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    cv::resize(ABflow,ABflow,cv::Size(),1/flowscale,1/flowscale,INTER_CUBIC);
                    BAflow=BAflow*1/flowscale;
                    ABflow=ABflow*1/flowscale;
                    cout <<"resizing optical flow : " << 1/flowscale << " [" << BAflow.size[1] << "," << BAflow.size[0] << "]" << endl;
                }
                //writing flo files
                if (writeflo == 1) {
                    sprintf(BAflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Bframe,Aframe);
                    sprintf(ABflowfile,"%s/%s_%04d_%04d.flo",outputdir,framename,Aframe,Bframe);
                    writeOpticalFlow(BAflowfile,BAflow);
                    writeOpticalFlow(ABflowfile,ABflow);
                }
            }
            //warping with flow A -> B
            mapx = cv::Mat::zeros(ABflow.size(), CV_32FC1);
            mapy = cv::Mat::zeros(ABflow.size(), CV_32FC1);
            for (int y = 0; y < ABflow.rows; ++y)
                {
                for (int x = 0; x < ABflow.cols; ++x)
                    {
                    Vec2f of = ABflow.at<Vec2f>(y, x);
                    mapx.at<float>(y, x) = (float)x - (of.val[0]*frac);
                    mapy.at<float>(y, x) = (float)y - (of.val[1]*frac);
                    }
                }
            cout << "warping Aframe" << endl;
            cv::remap(imageA,warpA,mapx,mapy,INTER_CUBIC,BORDER_REPLICATE);
            //warping with flow B -> A
            mapx = cv::Mat::zeros(BAflow.size(), CV_32FC1);
            mapy = cv::Mat::zeros(BAflow.size(), CV_32FC1);
            for (int y = 0; y < BAflow.rows; ++y)
                {
                for (int x = 0; x < BAflow.cols; ++x)
                    {
                    Vec2f of = BAflow.at<Vec2f>(y, x);
                    mapx.at<float>(y, x) = (float)x - (of.val[0]*(1.0-frac));
                    mapy.at<float>(y, x) = (float)y - (of.val[1]*(1.0-frac));
                    }
                }
            cout << "warping Bframe" << endl;
            cv::remap(imageB,warpB,mapx,mapy,INTER_CUBIC,BORDER_REPLICATE);
            //final mix
            cv::addWeighted(warpA,1.0-frac,warpB,frac,0.0,finalframe,-1);
            cout << "mixing A/B" << endl;
        }
        else {
            cout << "no optical flow : blending A/B" << endl;
            cv::addWeighted(imageA,1.0-frac,imageB,frac,0.0,finalframe,-1);
        }
        if (startatframe1 == 1) {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i-loop_start+1,extension);
        } else {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i,extension);
        }
        cout << "writing final frame : " << outputimage << endl;
        cv::imwrite(outputimage,finalframe);
        time = ((double) getTickCount() - startTick) / getTickFrequency();
        printf("Time [s]: %.3f\n", time);
        timecount++;
        averagetime+=time;
        cout << endl;
    }
    
    //no flow , just copy
    for (int i = transition_end; i <= loop_end; i++) {
        int Bframe=(2*loop_start)-i+loop_offset;
        sprintf(Bimage,"%s/%s.%04d.%s",inputdir,framename,Bframe,extension);
        cout << "processing loop frame : " << i-loop_start+1 << endl;
        cout <<"reading : " << Bimage << endl;
        imageB= imread(Bimage,IMREAD_COLOR);
        if (startatframe1 == 1) {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i-loop_start+1,extension);
        } else {
            sprintf(outputimage,"%s/%s_loop_m%d_%d_%d_%d.%04d.%s",outputdir,framename,flowmethod,loop_start,loop_end,loop_half_window,i,extension);
        }
        cout << "writing final frame : " << outputimage << endl;
        cv::imwrite(outputimage,imageB);
        cout << endl;
    }
    
}
