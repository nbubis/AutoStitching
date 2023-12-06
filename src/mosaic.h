#include "alignment.h"

class MosaicCreator
{
	public:
		MosaicCreator(ImageAligner & imageAligner, float resizedFactorForMosaic, bool blendImages);
		void saveMosaicImage(std::string outputPath);

	private:
		cv::Mat _mosaicImage;
		void setMosaicSize();
		void setImageWarpParams();
		void warpNonBlended(int i);
		void warpBlended(int i);

	private:
		const int _borderSize = 600;
		ImageAligner _imageAligner;
		cv::Mat _blendedMask;
		bool _blendImages;
		float _resizedFactorForMosaic;
		int _mosaicRowNum, _mosaicColNum, _xMosaicBias, _yMosaicBias;
		std::vector<cv::Rect> _warpedImageRects;
		std::vector<cv::Mat> _scaledHomographies;
		std::vector<cv::Mat> _biasedHomographies;
		cv::detail::MultiBandBlender _blender;
		std::vector<float> _distancesFromCenter;
};