#include "classifier.h"

#include <sys/time.h>
#include <fstream>

#include <boost/filesystem.hpp>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include <string>
#include <vector>
#include "ristretto/base_ristretto_layer.hpp"



int main(int argc, char** argv) {
  if (argc != 7)
  {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel quantization_param" << std::endl;
    return -1;
  }

  struct timeval begin, end;
  double elapsed;

  ::google::InitGoogleLogging(argv[0]);
#ifdef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

  std::string model_file   = argv[1];
  std::string trained_file = argv[2];
  std::string param_file = argv[3];
  std::string data_folder = argv[4];
  int num_faults = ::atoi(argv[5]);
  int MAX_ITER = ::atoi(argv[6]);
  std::cout << "[I] Testing on " << MAX_ITER << " images" << std::endl;
  std::string mean_file = "/home/aiembed/date-iccd-paper/testset/imagenet_mean.binaryproto";
  std::string labels_filename = "/opt/caffe/ristretto/data/ilsvrc12/val.txt";
  std::fstream labels_fstream;
  std::map<std::string, int> labels;
  labels_fstream.open(labels_filename, std::fstream::in | std::fstream::app);
  std::string line;
  int i = 0;
  gettimeofday(&begin, NULL);
  while( std::getline(labels_fstream, line) )
  {
    std::istringstream token_stream(line);
    std::string id_token;
    std::getline(token_stream, id_token, ' ');
    std::string key = id_token;
    std::getline(token_stream, id_token, ' ');
    int id = ::atoi(id_token.c_str());
    labels[key] =  id;
    i ++;
  }
  labels_fstream.close();
  gettimeofday(&end, NULL);
  elapsed = (end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_usec - begin.tv_usec);
  std::cout << "[I] Finished parsing idx_cls file in " << elapsed << " µs." << std::endl;

  gettimeofday(&begin, NULL);
  Classifier classifier(model_file, trained_file, mean_file, param_file);
  gettimeofday(&end, NULL);
  elapsed = (end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_usec - begin.tv_usec);
  std::cout << "[I] Network loaded in " << elapsed << " µs." << std::endl;

  std::cout << "[I] Injecting faults ..." << std::endl;
  gettimeofday(&begin, NULL);
  for (int i = 0 ; i < num_faults ; ++ i )
    classifier.InjectRandomSEU();
  gettimeofday(&end, NULL);
  elapsed = (end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_usec - begin.tv_usec);
  std::cout << "[I] Injection finished in " << elapsed << " µs." << std::endl;
  
  i = -1;
  int count = 0;
  for (boost::filesystem::directory_iterator itr(data_folder); itr != boost::filesystem::directory_iterator() ; ++ itr)
  {
    if (++ count > MAX_ITER)
      break;
    //boost::filesystem::directory_iterator itr(data_folder);
    std::string file = data_folder;
    file.append("/");
    file.append(itr->path().filename().string());
    if (boost::filesystem::is_regular_file(itr->status()))
    {
      i ++;
      //std::cout << " [" << boost::filesystem::file_size(itr->path()) << "]" << std::endl;
      gettimeofday(&begin, NULL);
      cv::Mat img = cv::imread(file, -1);
      gettimeofday(&end, NULL);
      elapsed = (end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_usec - begin.tv_usec);
      if (img.empty())
      {
        std::cout << "[E] Unable to decode image " << file << ". Skipping." << std::endl;
        //continue;
      }
      else
      {
        std::cout << "[I] Image " << file << " decoded in " << elapsed << " µs." << std::endl;
      }
      //CHECK(!img.empty()) << "Unable to decode image " << file;

      gettimeofday(&begin, NULL);
      std::vector<Prediction> predictions_before = classifier.Classify(img);
      gettimeofday(&end, NULL);
      elapsed = (end.tv_sec - begin.tv_sec) * 1000000 + (end.tv_usec - begin.tv_usec);

      std::cout << "[I] Precited: " << predictions_before[0].first << " for image " << file << " in " << elapsed << " µs." << std::endl;

      if (predictions_before[0].first == labels[itr->path().filename().string()])
        std::cout << "[D] Correct prediction with " << predictions_before[0].second << " accuracy" << std::endl;
      else
        std::cout << "[D] Predicted " << predictions_before[0].first << " with " << predictions_before[0].second << " accuracy while it should be " << labels[itr->path().filename().string()] << std::endl;

      // PRINT TOP 5 PREDICTIONS
      //std::cout << "Predictions for " << file << " :" << std::endl;
      //for(int i = 0 ; i < 5 ; ++ i )
      //{
      //  std::cout << predictions_before[i].first << ":" << predictions_before[i].second << std::endl;
      //}
    }
  }
  
  return 0;
}
