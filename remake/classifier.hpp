#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include "ristretto/base_ristretto_layer.hpp"

template<typename T>
class classifier {
  public:
    classifier(const std::string& model_file, const std::string& weights_file, const std::string& mean_file, const std::string& param_file) {

#ifdef CPU_ONLY
      caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
      caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

      srand(time(NULL));
      net_.reset(new caffe::Net<T>(model_file, caffe::TEST));
      net_->CopyTrainedLayersFrom(trained_file);
      std::cout << "[i] initialized classifier" << std::endl;
    };

    ~classifier() {};

 private:
  std::shared_ptr<caffe::Net<T> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;

  int net_params_size = 0;
  int net_blobs_size = 0;
  std::vector<int> params_size;
  std::vector<int> blobs_size;
};
