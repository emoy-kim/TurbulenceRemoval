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

#include <OpenCVLinker.h>

using namespace std;
using namespace cv;

#define REAL_PART    0
#define COMPLEX_PART 1

class TurbulenceRemoval
{
   struct Parameters
   {
      Mat D;
      Mat Factor;

      Parameters() = default;
   };
   Parameters TurbulenceRemover;

   bool NeedToReduction;
   uint FrameNumToInitialize;
   uint InitializationTiming;

   Size AnalysisFrameSize;
   int RowPadSize, ColPadSize;
   Mat HSVFrame;
   vector<vector<Mat>> UpdatedFrequencies;
   vector<float> InitialGradientNorms;

   void extractTargetChannel(Mat& target, const Mat& bgr_frame);
   void splitFourier(vector<Mat>& frequencies, const Mat& bgr_frame);
   float calculateGradientNorm(const vector<Mat>& fourier) const;
   void setFourierAndGradientNorm(const Mat& frame, const int& index_to_set);
   void initialize(const Mat& frame);

   float updateHeatFlows(const uint& frame_index);
   void suppressTurbulenceOnFFT();
   void extractMaxFrequenciesOnly(vector<Mat>& frequencies);
   void extractRealPartFromInverseFourier(Mat& heathaze_removed, const vector<Mat>& frequencies) const;
   void createComposedColorFrame(Mat& composed, const Mat& heathaze_removed, const Mat& bgr_frame) const;

   void removeOldParameters();


public:
   TurbulenceRemoval(const uint& frame_num_to_initialize = 3, const bool& need_to_reduction = false);

   void removeTurbulence(Mat& turbulence_removed, const Mat& frame);
};