/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is based on [1] which is coded by MATLAB.
 * In [1], the origin code is a offline version, but it is refactored and modified 
 * as an online version using parallel processing.
 * 
 * [1] https://sites.google.com/site/louyifei/research/turbulence
 * 
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

#define REAL_PART    0
#define COMPLEX_PART 1

class TurbulenceRemoval
{
public:
   TurbulenceRemoval(const uint& frame_num_to_initialize = 3, const bool& need_to_reduction = false);
   ~TurbulenceRemoval() = default;

   void removeTurbulence(cv::Mat& turbulence_removed, const cv::Mat& frame);

private:
   struct Parameters
   {
      cv::Mat D;
      cv::Mat Factor;

      Parameters() = default;
   };
   Parameters TurbulenceRemover;

   bool NeedToReduction;
   uint FrameNumToInitialize;
   uint InitializationTiming;

   cv::Size AnalysisFrameSize;
   int RowPadSize, ColPadSize;
   cv::Mat HSVFrame;
   std::vector<std::vector<cv::Mat>> UpdatedFrequencies;
   std::vector<float> InitialGradientNorms;

   void extractTargetChannel(cv::Mat& target, const cv::Mat& bgr_frame);
   void splitFourier(std::vector<cv::Mat>& frequencies, const cv::Mat& bgr_frame);
   float calculateGradientNorm(const std::vector<cv::Mat>& fourier) const;
   void setFourierAndGradientNorm(const cv::Mat& frame, const int& index_to_set);
   void initialize(const cv::Mat& frame);

   float updateHeatFlows(const uint& frame_index);
   void suppressTurbulenceOnFFT();
   void extractMaxFrequenciesOnly(std::vector<cv::Mat>& frequencies);
   void extractRealPartFromInverseFourier(cv::Mat& heathaze_removed, const std::vector<cv::Mat>& frequencies) const;
   void createComposedColorFrame(cv::Mat& composed, const cv::Mat& heathaze_removed, const cv::Mat& bgr_frame) const;

   void removeOldParameters();
};