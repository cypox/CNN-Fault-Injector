#include "classifier.h"
#include <cstdlib>

#include <boost/dynamic_bitset.hpp>

template <typename Dtype>
void Trim2FixedPoint_cpu(Dtype* data, const int cnt, const int bit_width, int fl)
{
  for (int index = 0; index < cnt; ++index)
  {
    Dtype before = data[index];
    // Saturate data
    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    data[index] /= pow(2, -fl);
    data[index] = round(data[index]); // NEAREST
    //data[index] = floor(data[index] + RandUniform_cpu()); // STOCHASTIC
    data[index] *= pow(2, -fl);

    // std::cout << "Before: " << before << " -- After : " << data[index] << std::endl;
  }
}

Classifier::Classifier(const std::string& model_file,
                       const std::string& trained_file,
                       const std::string& mean_file,
                       const std::string& param_file) {
#ifdef CPU_ONLY
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
  srand(time(NULL));

  /* Load the network. */
  net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);


  size_t params_count = net_->params().size();

  layer_params_sizes.reserve(params_count);
  layer_params_sizes.resize(params_count);

  for (int i = 0 ; i < params_count ; ++ i)
  {
    int p = net_->params()[i]->count();
    total_network_params += p;
    //std::cout << "[I] Layer " << i << " contains " << p << " params" << std::endl;
    layer_params_sizes[i] = p;
  }
  std::cout << "[I] Total network params " << total_network_params << std::endl;

  assert(params_count%2 == 0);
  layer_params.reserve(params_count/2);
  layer_params.resize(params_count/2);
  ParseParams(param_file, layer_params, params_count);

  /*
  // README
  // YOU DO NOT NEED TO QUANTIZE SINCE YOU USE RISTRETTO TO RUN INFERENCE
  // RISTRETTO DO IT ANYWAY IN THE CALL OF FORWARD
  // PARAMS AND OUTPUTS ARE QUANTIZED
  // IF YOU WANT TO MODIFY WEIGHTS HOWEVER, YOU HAVE TO CALL Trim2FixedPoint_cpu
  
  for ( int i = 0 ; i < params_count ; i += 2 )
  {
    boost::shared_ptr<caffe::Blob<float> > layer_param = net_->params()[i];
    boost::shared_ptr<caffe::Blob<float> > layer_bias = net_->params()[i+1];
    int p_count = layer_param->count();
    int b_count = layer_bias->count();

    int in_bw = layer_params[i/2][0];
    int in_fl = layer_params[i/2][1];
    int bw = layer_params[i/2][2];
    int fl = layer_params[i/2][3];
    int iw = (bw-fl-1)>0?(bw-fl-1):0;
    int dw = (fl>0)?fl:0;

    //std::cout << in_bw << " " << in_fl << " " << bw << " " << fl << " " << iw << " " << dw << std::endl;

    float* weights = layer_param->mutable_cpu_data();
    float* bias = layer_bias->mutable_cpu_data();
    Trim2FixedPoint_cpu<float>(weights, p_count, bw, fl);
    Trim2FixedPoint_cpu<float>(bias, b_count, bw, fl);
  }
  // YOU UNCOMMENT TO MANUALLY QUANTIZE. YOU DO NOT NEED IT FOR INFERENCE BUT YOU NEED IT IF YOU WORK ON WEIGHTS
  //*/

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(1000, output_layer->channels()) << "Number of labels is different from the output layer dimension.";

  int num_params = net_->params().size();
  for (int i = 0 ; i < num_params ; ++ i )
  {
      int layer_size = net_->params()[i]->count();
      net_params_size += layer_size;
      params_size.push_back(layer_size);
  }

  int num_blobs = net_->blobs().size();
  for (int i = 0 ; i < num_blobs ; ++ i )
  {
      int blob_size = net_->blobs()[i]->count();
      net_blobs_size += blob_size;
      blobs_size.push_back(blob_size);
  }
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(1000, N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(idx, output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const std::string& mean_file) {
  caffe::BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  caffe::Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /*
  // DEBUG - SHOWS VALUES OF A GIVEN LAYER (params/blobs)
  boost::shared_ptr<caffe::Blob<float> > intermediate_blob = net_->params()[0];
  float* intermediate_blob_data = intermediate_blob->mutable_cpu_data();
  int blob_size = intermediate_blob->count();
  std::cout << "Blob count: " << blob_size << std::endl;
  for (int i = 0 ; i < blob_size ; ++ i )
  {
    std::cout << intermediate_blob_data[i] << std::endl;
  }
  //*/

  /* Copy the output layer to a std::vector */
  caffe::Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
}


void Classifier::ParseParams(std::string quantization_file, std::vector<std::vector<int> >& layer_params, int params_count)
{
  std::ifstream quantization_stream(quantization_file);
  assert(quantization_stream.is_open());
  for ( int l = 0 ; l < params_count/2 ; ++ l )
  {
    std::string line;
    std::getline(quantization_stream, line);
    layer_params[l].reserve(4);
    layer_params[l].resize(4);
    std::istringstream token_stream(line);
    std::string token;
    std::string name;
    std::getline(token_stream, name, ';');
    std::getline(token_stream, token, ';');
    layer_params[l][0] = ::atoi(token.c_str());
    std::getline(token_stream, token, ';');
    layer_params[l][1] = ::atoi(token.c_str());
    std::getline(token_stream, token, ';');
    layer_params[l][2] = ::atoi(token.c_str());
    std::getline(token_stream, token, ';');
    layer_params[l][3] = ::atoi(token.c_str());
  }
}

void Classifier::InjectRandomSEU()
{
  int index = rand() % total_network_params;

  int param_count = net_->params().size();
  int layer_index = 0;
  int total_params_before = 0;
  int index_in_layer = 0;
  for ( int l = 0 ; l < param_count ; ++ l)
  {
    index_in_layer = index - total_params_before;
    total_params_before += layer_params_sizes[l];
    if (total_params_before > index)
    {
      break;
    }
    else
    {
      ++ layer_index;
    }
  }

  std::cout << "[I] Injecting in layer " << layer_index << " at index " << index_in_layer << " (" << index << ")" << std::endl;

  boost::shared_ptr<caffe::Blob<float> > current_layer_params = net_->params()[layer_index];
  float* layer = current_layer_params->mutable_cpu_data();
  int blob_size = current_layer_params->count();

  int in_bw = layer_params[layer_index/2][0];
  int in_fl = layer_params[layer_index/2][1];
  int bw = layer_params[layer_index/2][2];
  int fl = layer_params[layer_index/2][3];
  int iw = (bw-fl-1)>0?(bw-fl-1):0;
  int dw = (fl>0)?fl:0;

  std::cout << "[I] Layer " << layer_index << " contains " << blob_size << " parameters" << std::endl;
  std::cout << "[I] Layer " << layer_index << " is represented as " << in_bw << "," << in_fl << "," << bw << "," << fl << " (" << iw << "." << dw << ") fp" << std::endl;

  Trim2FixedPoint_cpu<float>(layer, blob_size, bw, fl);

  float fp_weight = layer[index_in_layer];
  boost::dynamic_bitset<> fp_weight_bin(32, *reinterpret_cast<unsigned long*>(&fp_weight));
  fp_weight = (fp_weight >= 0) ? fp_weight : -fp_weight;
  int integral = floor(fp_weight);
  int decimal = (fp_weight-integral) * pow(10, fl);

  boost::dynamic_bitset<> sign_bin(1);
  sign_bin[0] = (layer[index_in_layer] < 0);
  boost::dynamic_bitset<> integral_bin(iw, integral);
  boost::dynamic_bitset<> decimal_bin(dw);

  fp_weight -= integral;
  for ( int b = 1 ; b < fl+1 ; ++ b )
  {
    if (fp_weight >= pow(2,-b))
    {
      decimal_bin[fl-b] = 1;
      fp_weight -= pow(2, -b);
    }
    else
    {
      decimal_bin[fl-b] = 0;
    }
  }

  std::cout << "[I] STD " << layer[index_in_layer] << " FP " << "(" << sign_bin << ")" << integral_bin << "." << decimal_bin << std::endl; // DEBUG

  int bit_index = rand() % bw;
  std::cout << "[I] BIT FLIP AT POSITION " << bit_index << std::endl;

  float new_value = 0;
  if (bit_index == bw - 1 ) // sign bit injection
  {
    new_value = -layer[index_in_layer];
  }
  else
  {
    if (bit_index < iw)
    {
      integral_bin[bit_index] = !integral_bin[bit_index];
    }
    else
    {
      bit_index -= iw;
      decimal_bin[bit_index] = !decimal_bin[bit_index];
    }
    for (int i = 0 ; i < iw ; ++ i)
    {
      new_value += integral_bin[i] * pow(2, i);
    }
    for(int i = 1, j = dw - 1; j >= 0; ++ i, -- j)
    {
      new_value += decimal_bin[j] * pow(2, -i);
    }
  }
  
  std::cout << "[I] STD " << new_value << " FP " << "(" << sign_bin << ")" << integral_bin << "." << decimal_bin << std::endl; // DEBUG
  layer[index_in_layer] = new_value;
}

void Classifier::InjectSEUInLayer(double probability, int layer_index)
{
  boost::shared_ptr<caffe::Blob<float> > current_layer_params = net_->params()[layer_index];
  float* layer = current_layer_params->mutable_cpu_data();
  int blob_size = current_layer_params->count();
  int param_count = net_->params().size();
  int in_bw = layer_params[layer_index/2][0];
  int in_fl = layer_params[layer_index/2][1];
  int bw = layer_params[layer_index/2][2];
  int fl = layer_params[layer_index/2][3];
  int iw = (bw-fl-1)>0?(bw-fl-1):0;
  int dw = (fl>0)?fl:0;
  std::cout << "[I] Network contains " << param_count << " parameter vectors" << std::endl;
  std::cout << "[I] Layer " << layer_index << " contains " << blob_size << " parameters" << std::endl;
  std::cout << "[I] Layer " << layer_index << " is represented as " << in_bw << "," << in_fl << "," << bw << "," << fl << " (" << iw << "." << dw << ") fp" << std::endl;
  for (int i = 0 ; i < blob_size ; ++ i )
  {
    double f = (double)rand() / RAND_MAX;
    if (f < probability)
    {
      std::cout << "[H] Injected fault: was " << layer[i] << " and became 0" << std::endl;
      layer[i] = 0;
    }
  }
}

void Classifier::InjectSET(double probability, int blob_index)
{
  boost::shared_ptr<caffe::Blob<float> > layer_blob = net_->blobs()[blob_index];
  float* layer = layer_blob->mutable_cpu_data();
  int blob_size = layer_blob->count();
  for (int i = 0 ; i < blob_size ; ++ i )
  {
    double f = (double)rand() / RAND_MAX;
    if (f > probability)
    {
      layer[i] = 0;
    }
  }
}

