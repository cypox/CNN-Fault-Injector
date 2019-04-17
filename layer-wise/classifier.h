#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include "ristretto/base_ristretto_layer.hpp"


typedef std::pair<int, float> Prediction;

class Classifier {
 public:
  Classifier(const std::string& model_file, const std::string& trained_file, const std::string& mean_file, const std::string& param_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

  void InjectSET(double probability, int layer = -1);

  void InjectRandomSEU();

  void InjectSEUInLayer(int layer_index);

 private:
  void SetMean(const std::string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

  void ParseParams(std::string quantization_file, std::vector<std::vector<int> >& layer_params, int params_count);

 private:
  std::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;

  int net_params_size = 0;
  int net_blobs_size = 0;
  std::vector<int> params_size;
  std::vector<int> blobs_size;

  // Added for quantized networks
  std::vector<std::vector<int> > layer_params;
  std::vector<int> layer_params_sizes;
  int total_network_params = 0;
};

#endif // CLASSIFIER_H
