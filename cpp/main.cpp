#include <opencv2/opencv.hpp>

#include "svm.h"
#include "hog.h"

// #include "sr_common_inc.h"
// #include "ocl_runtime.h"
// #include "cv_resize.h"

// #if 0
// #undef PIP_PROFILE_TAG
// #undef PIP_PROFILE_TAG_CPU

// #define PIP_PROFILE_TAG(x)
// #define PIP_PROFILE_TAG_CPU(x)
// #endif

int main(/*const StMat& warpedBuf, const cv::Size& warpOutSize, int& dec_values, bool gpuflag*/)
{
    // PIP_PROFILE_BEGIN;
    cv::Mat img = cv::imread("../../input/rsz_warped.png", cv::IMREAD_GRAYSCALE);
    img.convertTo(img, CV_32FC1);
    img = img / 255.0f;
    cv::Mat imgfeature;
    int dec_values;

    // if(gpuflag){
    //     StMat img = StMat::zeros(StMat::GPU_BUF, 300, 400, CV_32FC1);
    //     resize_st(warpedBuf, img);
    //     // PIP_PROFILE_TAG("MarkerMode_Process gpu 1");

    //     getHogFeature_gpu(img, imgfeature);
    //     // PIP_PROFILE_TAG("MarkerMode_Process gpu 2");
    // }else{
    //     StMat rsz(warpedBuf.bufType(), 300, 400, warpedBuf.type());
    //     resize_st(warpedBuf, rsz);
    //     cv::Mat img(300, 400, CV_32FC1, cv::Scalar(0));
    //     rsz.asBuffer(StMat::CPU).refMat().convertTo(img, img.type());
    //     // PIP_PROFILE_TAG("MarkerMode_Process 1");

    //     getHogFeature(img, imgfeature);
    //     // PIP_PROFILE_TAG("MarkerMode_Process 2");
    // }

    // StMat rsz(warpedBuf.bufType(), 300, 400, warpedBuf.type());
    // resize_st(warpedBuf, rsz);
    // cv::Mat img(300, 400, CV_32FC1, cv::Scalar(0));
    // rsz.asBuffer(StMat::CPU).refMat().convertTo(img, img.type());
    // PIP_PROFILE_TAG("MarkerMode_Process 1");

    getHogFeature(img, imgfeature);
    // PIP_PROFILE_TAG("MarkerMode_Process 2");

    {
        std::cout << "------------------------" << std::endl;
        std::cout << "Hog Feature: " << std::endl;
        float *f = (float*)imgfeature.data;
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                std::cout << f[i * 8 + j] << " ";    
            }
            std::cout << std::endl;
        }
        std::cout << "------------------------" << std::endl << std::endl;
    }

    dec_values = -1;
    int feature_len = imgfeature.total();
    getSvmPredict((float*)imgfeature.data,
                    feature_len,
                    dec_values);
    // PIP_PROFILE_TAG("MarkerMode_Process 3");
    
    std::cout << "result: " << dec_values << std::endl;

    return 0;
}