#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>


class SeetaFaceIdentification{
public:
	SeetaFaceIdentification()		{}
	~SeetaFaceIdentification()		{}

	void start(const char* deploy_path, const char* trained_model_path, const char* mean_image_path)
	{
		#ifdef CPU_ONLY
		  caffe::Caffe::set_mode(caffe::Caffe::CPU);
		#else
		  caffe::Caffe::set_mode(caffe::Caffe::GPU);
		#endif
	    //caffe::Caffe::set_mode(caffe::Caffe::GPU);	

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
	boost::shared_ptr<caffe::Net<float> > net_;
};

void extractFeature_demo(const char* deploy_path, const char* trained_model_path, const char* mean_image_path, const char* image_path)
{
	SeetaFaceIdentification seetaFace;
	seetaFace.start(deploy_path, trained_model_path, mean_image_path);

	cv::Mat image = cv::imread(image_path);
        std::cout<<image_path<<std::endl;
	if (image.empty())
	{
		std::cout << "Load image failed!\n";
		return;		
	}
	/*
	* @brief if your image wiht background, you shoud detect face first and then crop face
	*/
	cv::imshow("face", image);
	cv::waitKey(1);
	seetaFace.setData(image);
	std::vector<float> feature;

    double accumulate_time = 0.0;
    int  loop_count = 1; //bigger than 1 for test elapsed time
    for(int i = 0; i < loop_count; ++i){
	int64 start = cv::getTickCount();
	seetaFace.execute(feature);
	accumulate_time += (cv::getTickCount() - start) / cv::getTickFrequency();
    }
	std::cout<<"Elapsed time: "<< accumulate_time / loop_count <<" ms."<<std::endl;
	std::copy(feature.begin(), feature.end(), std::ostream_iterator<float>(std::cout, "\t"));
    std::cout<<std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::cout<<"Usage: seetaIdentification <path to deploy> <path to caffemodel> <path to mean image> <path to image>"<<std::endl;
		return 0;
	}
	extractFeature_demo(argv[1], argv[2], argv[3], argv[4]);
    return 1;
}
