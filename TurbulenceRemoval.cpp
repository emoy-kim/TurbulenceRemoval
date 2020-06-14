#include "TurbulenceRemoval.h"

class ParallelMatrixMultiply : public cv::ParallelLoopBody
{
   cv::Mat& mul;
   const cv::Mat& mat1;
   const cv::Mat& mat2;

public:
   ParallelMatrixMultiply(cv::Mat& mul, const cv::Mat& mat1, const cv::Mat& mat2) : mul( mul ), mat1( mat1 ), mat2( mat2 ) {}

   void operator()(const cv::Range& range) const override
   {
      const int col_step = mul.cols;
      
      auto* mul_ptr = reinterpret_cast<float*>(mul.data) + range.start * col_step;
      const auto* row1_ptr = reinterpret_cast<float*>(mat1.data) + range.start * col_step;
      const auto* row2_ptr = reinterpret_cast<float*>(mat2.data) + range.start * col_step;

      for (int j = range.start; j < range.end; ++j) {
         for (int i = 0; i < mul.cols; ++i) {
            mul_ptr[i] = row1_ptr[i] * row2_ptr[i];
         }

         mul_ptr += col_step;
         row1_ptr += col_step;
         row2_ptr += col_step;
      }
   }
};

class ParallelCalculateGradientNorm : public cv::ParallelLoopBody
{
   cv::Mat& norm;
   const cv::Mat& real;
   const cv::Mat& complex;
   const cv::Mat& d;
   
public:
   ParallelCalculateGradientNorm(cv::Mat& norm, const cv::Mat& real, const cv::Mat& complex, const cv::Mat& d) : 
      norm( norm ), real( real ), complex( complex ), d( d ) {}

   void operator()(const cv::Range& range) const override
   {
      const int col_step = real.cols;
      
      auto* norm_ptr = reinterpret_cast<float*>(norm.data) + range.start * col_step;
      const auto* real_ptr = reinterpret_cast<float*>(real.data) + range.start * col_step;
      const auto* complex_ptr = reinterpret_cast<float*>(complex.data) + range.start * col_step;
      const auto* d_ptr = reinterpret_cast<float*>(d.data) + range.start * col_step;

      for (int j = range.start; j < range.end; ++j) {
         for (int i = 0; i < real.cols; ++i) {
            norm_ptr[i] = (real_ptr[i] * real_ptr[i] + complex_ptr[i] * complex_ptr[i]) * d_ptr[i];
         }

         norm_ptr += col_step;
         real_ptr += col_step;
         complex_ptr += col_step;
         d_ptr += col_step;
      }
   }
};

class ParallelUpdateHeatFlows : public cv::ParallelLoopBody
{
   std::vector<std::vector<cv::Mat>>& frequencies;
   const cv::Mat& factor;
   const float& gradient_norm;
   const uint& frame_index;

   void accumulate(
      cv::Mat& real, cv::Mat& complex, 
      const cv::Mat& prev_real, const cv::Mat& prev_complex, 
      const cv::Mat& next_real, const cv::Mat& next_complex, 
      float norm_factor, int row
   ) const
   {
      auto* real_ptr = real.ptr<float>(row);
      auto* complex_ptr = complex.ptr<float>(row);
      const auto* prev_real_ptr = prev_real.ptr<float>(row);
      const auto* prev_complex_ptr = prev_complex.ptr<float>(row);
      const auto* next_real_ptr = next_real.ptr<float>(row);
      const auto* next_complex_ptr = next_complex.ptr<float>(row);
      const auto* factor_ptr = factor.ptr<float>(row);

      for (int i = 0; i < real.cols; ++i) {
         const float gradient_factor = gradient_norm * factor_ptr[i] + 1.0f;
         real_ptr[i] = (gradient_factor * real_ptr[i] + (prev_real_ptr[i] + next_real_ptr[i])) * norm_factor;
         complex_ptr[i] = (gradient_factor * complex_ptr[i] + (prev_complex_ptr[i] + next_complex_ptr[i])) * norm_factor;
      }
   }

   void accumulateAtBoundary(
      cv::Mat& real, cv::Mat& complex, 
      const cv::Mat& boundary_real, const cv::Mat& boundary_complex, 
      float norm_factor, int row
   ) const
   {
      auto* real_ptr = real.ptr<float>(row);
      auto* complex_ptr = complex.ptr<float>(row);
      const auto* boundary_real_ptr = boundary_real.ptr<float>(row);
      const auto* boundary_complex_ptr = boundary_complex.ptr<float>(row);
      const auto* factor_ptr = factor.ptr<float>(row);

      for (int i = 0; i < real.cols; ++i) {
         const float gradient_factor = gradient_norm * factor_ptr[i] + 1.0f;
         real_ptr[i] = (gradient_factor * real_ptr[i] + boundary_real_ptr[i]) * norm_factor;
         complex_ptr[i] = (gradient_factor * complex_ptr[i] + boundary_complex_ptr[i]) * norm_factor;
      }
   }

public:
   ParallelUpdateHeatFlows(std::vector<std::vector<cv::Mat>>& frequencies, const cv::Mat& factor, float gradient_norm, uint frame_index) : 
      frequencies( frequencies ), factor( factor ), gradient_norm( gradient_norm ), frame_index( frame_index ) {}

   void operator()(const cv::Range& range) const override
   {
      cv::Mat& real = frequencies[frame_index][0];
      cv::Mat& complex = frequencies[frame_index][1];

      float norm_factor;
      if (frame_index == 0) {
         norm_factor = 0.5f;

         cv::Mat& next_real = frequencies[1][REAL_PART];
         cv::Mat& next_complex = frequencies[1][COMPLEX_PART];
         for (int j = range.start; j < range.end; ++j) {
            accumulateAtBoundary( real, complex, next_real, next_complex, norm_factor, j );
         }
      }
      else if (frame_index == frequencies.size() - 1) {
         norm_factor = 0.5f;

         cv::Mat& prev_real = frequencies[frame_index - 1][0];
         cv::Mat& prev_complex = frequencies[frame_index - 1][1];
         for (int j = range.start; j < range.end; ++j) {
            accumulateAtBoundary( real, complex, prev_real, prev_complex, norm_factor, j );
         }
      }
      else {
         norm_factor = 1.0f / 3.0f;

         cv::Mat& prev_real = frequencies[frame_index - 1][0];
         cv::Mat& prev_complex = frequencies[frame_index - 1][1];
         cv::Mat& next_real = frequencies[frame_index + 1][0];
         cv::Mat& next_complex = frequencies[frame_index + 1][1];
         for (int j = range.start; j < range.end; ++j) {
            accumulate( real, complex, prev_real, prev_complex, next_real, next_complex, norm_factor, j );
         }
      }
   }
};


TurbulenceRemoval::TurbulenceRemoval(const uint& frame_num_to_initialize, const bool& need_to_reduction) : 
   NeedToReduction( need_to_reduction ), FrameNumToInitialize( frame_num_to_initialize ), 
   InitializationTiming( 0 ), RowPadSize( 0 ), ColPadSize( 0 ), 
   UpdatedFrequencies( frame_num_to_initialize, std::vector<cv::Mat>(2) ), InitialGradientNorms( frame_num_to_initialize )
{
}

void TurbulenceRemoval::extractTargetChannel(cv::Mat& target, const cv::Mat& bgr_frame)
{
   std::vector<cv::Mat> channels;
   cvtColor( bgr_frame, HSVFrame, cv::COLOR_BGR2HSV );
   
   split( HSVFrame, channels );

   target = channels[2].clone(); 
}

void TurbulenceRemoval::splitFourier(std::vector<cv::Mat>& frequencies, const cv::Mat& bgr_frame)
{
   cv::Mat target_channel, padded_target_channel;
   extractTargetChannel( target_channel, bgr_frame );
   cv::copyMakeBorder( 
      target_channel, 
      padded_target_channel, 
      0, RowPadSize, 
      0, ColPadSize, 
      cv::BORDER_REPLICATE
   );
   padded_target_channel.convertTo( padded_target_channel, CV_32FC1 );

   cv::Mat fourier;
   cv::merge( std::vector<cv::Mat>{ padded_target_channel, cv::Mat::zeros(padded_target_channel.size(), CV_32FC1) }, fourier );

   dft( fourier, fourier );
   
   split( fourier, frequencies );
}

float TurbulenceRemoval::calculateGradientNorm(const std::vector<cv::Mat>& fourier) const
{
   cv::Mat gradient_norm(fourier[0].size(), CV_32FC1);
   parallel_for_( 
      cv::Range(0, fourier[0].rows), 
      ParallelCalculateGradientNorm(gradient_norm, fourier[REAL_PART], fourier[COMPLEX_PART], TurbulenceRemover.D) 
   );
   return static_cast<float>(abs( sum( gradient_norm )[0] ));
}

void TurbulenceRemoval::setFourierAndGradientNorm(const cv::Mat& frame, const int& index_to_set)
{
   cv::Mat resized;
   resize( frame, resized, AnalysisFrameSize );
   splitFourier( UpdatedFrequencies[index_to_set], resized );
   InitialGradientNorms[index_to_set] = calculateGradientNorm( UpdatedFrequencies[index_to_set] );
}

void TurbulenceRemoval::initialize(const cv::Mat& frame)
{
   AnalysisFrameSize.width = NeedToReduction ? frame.cols / 2 : frame.cols;
   AnalysisFrameSize.height = NeedToReduction ? frame.rows / 2 : frame.rows;
   RowPadSize = cv::getOptimalDFTSize( AnalysisFrameSize.height ) - AnalysisFrameSize.height;
   ColPadSize = cv::getOptimalDFTSize( AnalysisFrameSize.width ) - AnalysisFrameSize.width;

   TurbulenceRemover.D.create( AnalysisFrameSize.height + RowPadSize, AnalysisFrameSize.width + ColPadSize, CV_32FC1 );
   const auto pi_over_width = static_cast<const float>(CV_PI / TurbulenceRemover.D.cols);
   const auto pi_over_height = static_cast<const float>(CV_PI / TurbulenceRemover.D.rows);
   for (int j = 0; j < TurbulenceRemover.D.rows; ++j) {
      auto* d_ptr = TurbulenceRemover.D.ptr<float>(j);
      const float sin_square = pow( sin( j * pi_over_height ), 2 );
      for (int i = 0; i < TurbulenceRemover.D.cols; ++i) {
         d_ptr[i] = pow( sin( i * pi_over_width ), 2 ) + sin_square;
      }
   }

   const cv::Mat factor1 = 1.0f / (1.0f + 20.0f * TurbulenceRemover.D);
   const cv::Mat factor2 = 4.0f * TurbulenceRemover.D;
   TurbulenceRemover.Factor.create( TurbulenceRemover.D.size(), TurbulenceRemover.D.type() );
   parallel_for_( cv::Range(0, TurbulenceRemover.Factor.rows), ParallelMatrixMultiply(TurbulenceRemover.Factor, factor1, factor2) );

   setFourierAndGradientNorm( frame, 0 );
}

float TurbulenceRemoval::updateHeatFlows(const uint& frame_index)
{
   const float gradient_norm = calculateGradientNorm( UpdatedFrequencies[frame_index] ) / InitialGradientNorms[frame_index] - 5.0f;
   parallel_for_( 
      cv::Range(0, UpdatedFrequencies[frame_index][REAL_PART].rows), 
      ParallelUpdateHeatFlows(UpdatedFrequencies, TurbulenceRemover.Factor, -gradient_norm, frame_index)
   );
   return gradient_norm;
}

void TurbulenceRemoval::suppressTurbulenceOnFFT()
{
   std::vector<float> updated_gradient_norms(FrameNumToInitialize, 0.0);
   for (uint it = 0; it < 3; ++it) {
      for (uint n = 0; n < UpdatedFrequencies.size(); ++n) {
         updated_gradient_norms[it] += updateHeatFlows( n );
      }
      if (it >= 1 && abs( updated_gradient_norms[it] - updated_gradient_norms[it - 1] ) < 5e-3f) break;
   }
}

void TurbulenceRemoval::extractMaxFrequenciesOnly(std::vector<cv::Mat>& frequencies)
{
   cv::Mat filtered_real( UpdatedFrequencies[0][REAL_PART].size(), CV_32FC1 );
   cv::Mat filtered_complex( UpdatedFrequencies[0][REAL_PART].size(), CV_32FC1 );
   for (int j = 0; j < UpdatedFrequencies[0][REAL_PART].rows; ++j) {
      auto* filtered_real_ptr = filtered_real.ptr<float>(j);
      auto* filtered_complex_ptr = filtered_complex.ptr<float>(j);

      std::vector<const float*> real_ptrs(UpdatedFrequencies.size());
      std::vector<const float*> complex_ptrs(UpdatedFrequencies.size());
      for (uint n = 0; n < UpdatedFrequencies.size(); ++n) {
         real_ptrs[n] = UpdatedFrequencies[n][REAL_PART].ptr<float>(j);
         complex_ptrs[n] = UpdatedFrequencies[n][COMPLEX_PART].ptr<float>(j);
      }

      for (int i = 0; i < UpdatedFrequencies[0][REAL_PART].cols; ++i) {
         float max_magnitude = 0.0f;
         float sum_real = 0.0f;
         float sum_complex = 0.0f;
         for (uint n = 0; n < real_ptrs.size(); ++n) {
            const float magnitude = real_ptrs[n][i] * real_ptrs[n][i] + complex_ptrs[n][i] * complex_ptrs[n][i];
            if (magnitude > max_magnitude) max_magnitude = magnitude;
            sum_real += real_ptrs[n][i];
            sum_complex += complex_ptrs[n][i];
         }

         float squared_sum = sum_real * sum_real + sum_complex * sum_complex;
         if (squared_sum < 1e-7f) squared_sum = 1e-7f;
         const float norm_factor = sqrt( max_magnitude / squared_sum );
         filtered_real_ptr[i] = sum_real * norm_factor;
         filtered_complex_ptr[i] = sum_complex * norm_factor;
      }
   }

   frequencies.clear();
   frequencies = { filtered_real, filtered_complex };
}

void TurbulenceRemoval::extractRealPartFromInverseFourier(cv::Mat& heathaze_removed, const std::vector<cv::Mat>& frequencies) const
{
   cv::Mat fourier;
   merge( frequencies, fourier );

   idft( fourier, heathaze_removed, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE );

   heathaze_removed.convertTo( heathaze_removed, CV_8UC1 );

   heathaze_removed = heathaze_removed(cv::Rect(0, 0, heathaze_removed.cols - ColPadSize, heathaze_removed.rows - RowPadSize));
}

void TurbulenceRemoval::createComposedColorFrame(cv::Mat& composed, const cv::Mat& heathaze_removed, const cv::Mat& bgr_frame) const
{
   std::vector<cv::Mat> channels;
   split( HSVFrame, channels );

   merge( std::vector<cv::Mat>{ channels[0], channels[1], heathaze_removed }, composed );
   
   cv::Mat resized_heathaze_removed;
   resize( composed, composed, bgr_frame.size() );

   cvtColor( composed, composed, cv::COLOR_HSV2BGR );
}

void TurbulenceRemoval::removeOldParameters()
{
   while (UpdatedFrequencies.size() >= FrameNumToInitialize) {
      UpdatedFrequencies.erase( UpdatedFrequencies.begin() );
      InitialGradientNorms.erase( InitialGradientNorms.begin() );
   }
   
   UpdatedFrequencies.emplace_back( std::vector<cv::Mat>(2) );
   InitialGradientNorms.emplace_back();
}

void TurbulenceRemoval::removeTurbulence(cv::Mat& turbulence_removed, const cv::Mat& frame)
{
   if (InitializationTiming < FrameNumToInitialize) {
      if (InitializationTiming == 0) initialize( frame );
      else setFourierAndGradientNorm( frame, InitializationTiming );
      turbulence_removed = frame.clone();
      InitializationTiming++;
      return;
   }

   removeOldParameters();

   setFourierAndGradientNorm( frame, FrameNumToInitialize - 1 );

   suppressTurbulenceOnFFT();

   std::vector<cv::Mat> filtered;
   extractMaxFrequenciesOnly( filtered );

   cv::Mat heathaze_removed;
   extractRealPartFromInverseFourier( heathaze_removed, filtered );

   createComposedColorFrame( turbulence_removed, heathaze_removed, frame );
}