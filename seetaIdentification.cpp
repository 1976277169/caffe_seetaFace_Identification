#include <caffe/caffe.hpp>
#include <opencv.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
/*
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file);
  std::vector<float> Predict(const cv::Mat& img);
  void printModelParam();
 private:

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file) {
//#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
//#else
//  Caffe::set_mode(Caffe::GPU);
//#endif

  /// Load the network. 
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  //Blob<float>* output_layer = net_->output_blobs()[0];
  //CHECK_EQ(labels_.size(), output_layer->channels())
  //  << "Number of labels is different from the output layer dimension.";
}

void Classifier::printModelParam()
{
	std::FILE *fout = std::fopen("trained_model.bin", "wb");
	if (!fout) return;

	std::vector < shared_ptr<caffe::Layer<float>>> lay = net_->layers();
	for (int i = 0; i < lay.size(); i++)
	{
		
		std::vector<shared_ptr<caffe::Blob<float>>> ps = lay[i]->blobs();
		const char *name = lay[i]->type();
		
		if (!std::strcmp(name, "Convolution"))
		{
			int pad = 1;
			int stride = 1;
			//int kernel_size = 5;
			
			//int output = sh[0];
			fwrite(&pad, sizeof(int), 1, fout);
			fwrite(&stride, sizeof(int), 1, fout);
		}
		else if (!std::strcmp(name, "Pooling"))
		{
			int pad = 0;
			int stride = 2;
			int kernel_size = 2;
			fwrite(&pad, sizeof(int), 1, fout);
			fwrite(&stride, sizeof(int), 1, fout);
			fwrite(&kernel_size, sizeof(int), 1, fout);
		}
		std::cout << name << std::endl;
		for (int j = 0; j < ps.size(); j++)
		{
			std::vector<int> sh = ps[j]->shape();
			fwrite(&sh[0], sizeof(int), sh.size(), fout);
			float *data = ps[j]->mutable_cpu_data();
			int sz = ps[j]->count();
			std::fwrite(data, sizeof(float), sz, fout);
		}
	}
	std::fclose(fout);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  // Forward dimension change to all layers. 
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  input_layer->scale_data(0.00392156862f);

  //std::cout << "\n";
  //std::copy(input_layer->cpu_data(), input_layer->cpu_data() + input_layer->count(), std::ostream_iterator<float>(std::cout, "\t"));
  net_->Forward();

  // Copy the output layer to a std::vector 
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}


void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

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
  // Convert the input image to the input image format of the network. /
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

  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

//void funcprocessing(std::string &model_file, std::string &trained_file) {
//
//  
//
//  cv::Mat img = cv::imread("");
//  if (img.empty()) std::cout << "Invalid image!" << std::endl;
//  
//
//  cv::vector<float> res = classifier.Predict(img);
//  ///* Print the top N predictions. 
//  //for (size_t i = 0; i < predictions.size(); ++i) {
//  //  Prediction p = predictions[i];
//  //  std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
//  //            << p.first << "\"" << std::endl;
//  }

int splitStr(std::string &path, char buf[])
{
	int i = 0;
	while (buf[i] != '\t') i++;
	char *p = &buf[i + 1];
	buf[i] = '\0';
	path = std::string(buf);

	return std::atoi(p);
}

int main()
{
	std::string trained_file = "E:\\BaiduYunDownload\\vgg_face_caffe\\VGG_FACE.caffemodel";
	std::string model_file = "E:\\BaiduYunDownload\\vgg_face_caffe\\VGG_FACE_deploy.prototxt";
	Classifier classifier(model_file, trained_file);

	classifier.printModelParam();

	//std::string test_file = "E:/Face_alignment/Multi-task/Code/cnn-master/Dataset/MNIST/data/test_picImages/list.txt";
	//std::string root_dir = "E:/Face_alignment/Multi-task/Code/cnn-master/Dataset/MNIST/";
	std::string test_file = "E:/tHandDrawnShapeCollector/tHandDrawnShapeCollector/picImages/test.txt";
	std::string root_dir = "E:/tHandDrawnShapeCollector/tHandDrawnShapeCollector";
	std::string path;
	int accCount = 0, total = 0;
	std::ifstream fin(test_file);
	std::vector<float> result;
	char buf[260];
	double tt = 0.0;
	while (fin.getline(buf, 260))
	{
		int label = splitStr(path, buf);

		cv::Mat im = cv::imread(root_dir + path, 0);
		if (im.empty()) continue;

		total++;

		double t = double(cv::getTickCount());
		result.clear();
		result = classifier.Predict(im);
		tt += (double(cv::getTickCount()) - t)/cv::getTickFrequency();

		int idx = -1;
		float maxClass = -1.0f;
		for (int i = 0; i < result.size(); i++)			if (result[i] > maxClass) { maxClass = result[i], idx = i; }
		if (label == idx) accCount++;
		printf("Processed:\t%d\r", total);
	}
	printf("Average time:\t%lf\n", tt / total);
	printf("accCount:\t%d\ntotal:\t%d\n", accCount, total);
	printf("Accuracy:\t%f%%\n", float(100 * accCount) / total);
	system("pause");
	return 0;
}

//int main(int argc, char** argv) {
//  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
//}
*/

class SeetaFaceIdentification{
public:
	SeetaFaceIdentification()		{}
	~SeetaFaceIdentification()		{}

	void start(const char* deploy_path, const char* trained_model_path, const char* mean_image_path)
	{
		/*
		#ifdef CPU_ONLY
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
		#else
		  caffe::Caffe::set_mode(caffe::Caffe::GPU);
		#endif
		*/
		caffe::Caffe::set_mode(caffe::Caffe::CPU);

		net_.reset(new caffe::Net<float>(std::string(deploy_path), caffe::TEST));
		net_->CopyTrainedLayersFromBinaryProto(std::string(trained_model_path));

		CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

		read_mean_image(mean_image_path);
	}

	void setData(cv::Mat &image)
	{
		if (image.empty())
		{
			std::cout << "Invalid input image!\n";
			return;
		}

		cv::Mat bgrIm;
		if (image.channels() == 1)
			cv::cvtColor(image, bgrIm, CV_GRAY2BGR);
		else if (image.channels() == 4)
			cv::cvtColor(image, bgrIm, CV_BGRA2BGR);
		else if (image.channels() == 3)
			bgrIm = image;
		else
		{
			std::cout << "Unkown input Image type, The required type is CV_BGR!\n";
			return;
		}

		cv::Mat square_256_im;
		if (bgrIm.cols != 256 || bgrIm.rows != 256)
			cv::resize(bgrIm, square_256_im, cv::Size(256, 256));
		else
			square_256_im = bgrIm;
	
		cv::Mat float_im;
		if (square_256_im.type() != CV_32FC3)
			square_256_im.convertTo(float_im, CV_32FC3);
		else
			float_im = square_256_im;

		int W = float_im.cols, H = float_im.rows, C = float_im.channels(), stride = *float_im.step.p/sizeof(float);
		std::vector<float> temp(W*H*C);
		std::vector<float> temp1(W*H*C);
		permute_image(W, H, C, W, &temp[0], stride, (float *)float_im.data);

		subtract_mean(W, H, C, W, &temp1[0], &temp[0]);

		int cW, cH, cStride;
		pad_image(cW, cH, cStride, C, &temp[0], W, H, W, &temp1[0]);

		cv::Mat dis = cv::Mat(cH, cW, CV_32FC1, (float *)&temp[0], sizeof(float)*cW);

		caffe::Blob<float>* input_blob = net_->input_blobs()[0];
		int input_width = input_blob->width();
		int input_height = input_blob->height();
		int input_channels = input_blob->channels();

		if (input_width != cW || input_height != cH || input_channels != C)
		{
			std::cout << "Final input image to caffe don't match with deploy input parameters\n";
			return;
		}

		int single_image_size = input_channels * input_height * input_width;
		std::copy(&temp[0], &temp[0] + single_image_size, input_blob->mutable_cpu_data());
	}

	void execute(std::vector<float> &feat)
	{
		feat.clear();

		
		net_->Forward();

		caffe::Blob<float>* output_blob = net_->output_blobs()[0];
		int count = output_blob->count();
		feat.resize(count);
		std::copy(output_blob->cpu_data(), output_blob->cpu_data() + count, &feat[0]);
	}

private:
	void read_mean_image(const char* path)
	{
		std::ifstream ifs;
		ifs.open(path, std::ios::in);
		if (!ifs) return;

		float val = 0.0f;
		while (!ifs.eof())
		{
			ifs >> val;
			mean_image_.push_back(val);
		}
		ifs.close();
	}

	/*
	* @brief crop iamge 256*256 to 228*228
	*        NOTE: width_pad and height_pad only negative integer, otherwise, you will get an error
	*/
	void pad_image(int &oW, int &oH, int &oStride, int C, float *pOut, int iW, int iH, int iStride, float *pIn, int width_pad = -14, int height_pad = -14)
	{
		oW = iW + 2 * width_pad;
		oH = iH + 2 * height_pad;
		oStride = oW;

		float *pO = pOut, *po;
		float *pI = pIn + (-height_pad) * iStride - width_pad, *pi;

		int iSpatial_size = iW * iH;
		int oSpatial_size = oW * oH;
		int i, j, k;
		for (k = 0; k < C; k++, pO += oSpatial_size, pI += iSpatial_size)
		{
			pi = pI, po = pO;
			for (j = 0; j < oH; ++j, pi += iStride, po += oStride)
			{
				std::memcpy(po, pi, sizeof(float)*oStride);
			}
		}
	}

	/*
	* @brief subtract mean image from source image
	*/
	void subtract_mean(int W, int H, int C, int Stride, float *pOut, float *pIn)
	{
		float *pO = pOut, *po;
		float *pI = pIn, *pi;
		float *pM = &mean_image_[0], *pm;

		int spatial_size = W * H;
		int i, j, k;
		for (k = 0; k < C; k++, pO += spatial_size, pI += spatial_size, pM += spatial_size)
		{
			po = pO, pi = pI, pm = pM;
			for (j = 0; j < H; ++j, po += W, pi += W, pm += W)
			{
				for (i = 0; i < W; ++i)
					po[i] = pi[i] + pm[i];
			}
		}
	}

	/**
	 * @brief permute opencv format image(width * height * channels) to Caffe format image(channels * width * height)
	*/
	void permute_image(int W, int H, int C, int oStride, float *pOut, int iStride, float *pIn)
	{
		float *pB = pOut, *po;
		float *pI = pIn, *pi;
		
		int spatial_size = W*H;
		float *pG = pB + spatial_size, *pR = pG + spatial_size;
		int i, j;
		for (j = 0; j < H; ++j, pI += iStride, pB += W, pG += W, pR += W)
		{
			pi = pI;
			for (i = 0; i < W; ++i, pi += C)
			{
				pB[i] = pi[0];
				pG[i] = pi[1];
				pR[i] = pi[2];
			}
		}
	}

	std::vector<float> mean_image_;
	std::shared_ptr<caffe::Net<float> > net_;
};

void extractFeature_demo(const char* deploy_path, const char* trained_model_path, const char* mean_image_path)
{
	SeetaFaceIdentification seetaFace;
	seetaFace.start(deploy_path, trained_model_path, mean_image_path);

	cv::Mat image = cv::imread("E:/CODE/caffe-app/caffe-app/NF_200001_001.jpg");
	if (image.empty())
		std::cout << "Load image failed!\n";

	/*
	* @brief if your image wiht background, you shoud detect face first and then crop face
	*/
	
	seetaFace.setData(image);
	std::vector<float> feature;
	seetaFace.execute(feature);
	std::copy(feature.begin(), feature.end(), std::ostream_iterator<float>(std::cout, "\t"));
}

void main()
{
	extractFeature_demo("E:/CODE/caffe-app/caffe-app/deploy.prototxt", "E:/CODE/caffe-app/caffe-app/seetaFace_identification.caffemodel", "E:/CODE/caffe-app/caffe-app/mean_image.txt");
}