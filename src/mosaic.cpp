#include <numeric>
#include <execution>
#include "mosaic.h"

MosaicCreator::MosaicCreator(ImageAligner & imageAligner, float resizedFactorForMosaic, bool blendImages) : 
    _imageAligner(imageAligner), _resizedFactorForMosaic(resizedFactorForMosaic), _blendImages(blendImages)
{
    std::cout << "Warping images ..." << std::endl;
    _scaledHomographies.resize(_imageAligner._imgNum);
    _biasedHomographies.resize(_imageAligner._imgNum);
    _warpedImageRects.resize(_imageAligner._imgNum);
	_distancesFromCenter.resize(_imageAligner._imgNum);

    setMosaicSize();
    setImageWarpParams();
    _mosaicImage = cv::Mat(_mosaicRowNum, _mosaicColNum, CV_8UC4, Scalar(0, 0, 0, 0));

    if (blendImages) {
		_blendedMask = cv::Mat(_mosaicRowNum + 2*_borderSize / _resizedFactorForMosaic, _mosaicColNum + 2*_borderSize / _resizedFactorForMosaic, CV_8UC1, Scalar(0));
        _blender = cv::detail::MultiBandBlender(0, 9);
	    _blender.prepare(cv::Rect(0, 0, _blendedMask.cols, _blendedMask.rows));
    }
}


void MosaicCreator::setMosaicSize() 
{

	std::vector<cv::Point2d> warpedCornersAll, warpedCorners;

	for (int i = 0; i < _imageAligner._imgNum; i++) 
	{
		cv::Mat homography = _imageAligner._alignModelList[_imageAligner.findVisitIndex(i)].clone();
		homography.row(0) = homography.row(0) / _resizedFactorForMosaic;
		homography.row(1) = homography.row(1) / _resizedFactorForMosaic;
		_scaledHomographies[i] = homography;

        auto imageSize = _imageAligner._imgSizeList[i];
		std::vector<cv::Point2d> imageCorners{{0.0, 0.0}, {imageSize.width - 1.0, 0.0}, 
			{imageSize.width - 1.0, imageSize.height - 1.0}, {0.0, imageSize.height - 1.0}};
		cv::perspectiveTransform(imageCorners, warpedCorners, _scaledHomographies[i]);
		warpedCornersAll.insert(std::end(warpedCornersAll), std::begin(warpedCorners), std::end(warpedCorners));
    };
	
	auto minmaxX = std::minmax_element(warpedCornersAll.begin(), warpedCornersAll.end(), 
		[](auto a, auto b){ return a.x < b.x; });
	auto minmaxY = std::minmax_element(warpedCornersAll.begin(), warpedCornersAll.end(), 
		[](auto a, auto b){ return a.y < b.y; });

	_xMosaicBias = -1.0 * minmaxX.first->x;
	_yMosaicBias = -1.0 * minmaxY.first->y;

	_mosaicColNum = minmaxX.second->x - minmaxX.first->x;
	_mosaicRowNum = minmaxY.second->y - minmaxY.first->y;

}


void MosaicCreator::setImageWarpParams() 
{
    for (int i = 0; i < _imageAligner._imgNum; i++)
    {
        auto imageSize = _imageAligner._imgSizeList[i];
        std::vector<cv::Point2d> imageCorners{{0.0, 0.0}, {imageSize.width - 1.0, 0.0}, 
            {imageSize.width - 1.0, imageSize.height - 1.0}, {0.0, imageSize.height - 1.0}};
        std::vector<cv::Point2d> warpedCorners;
        cv::perspectiveTransform(imageCorners, warpedCorners, _scaledHomographies[i]);

        auto imageMinmaxX = std::minmax_element(warpedCorners.begin(), warpedCorners.end(), 
            [](auto a, auto b){ return a.x < b.x; });
        auto imageMinmaxY = std::minmax_element(warpedCorners.begin(), warpedCorners.end(), 
            [](auto a, auto b){ return a.y < b.y; });

        int imageColNum = imageMinmaxX.second->x - imageMinmaxX.first->x;
        int imageRowNum = imageMinmaxY.second->y - imageMinmaxY.first->y;

        _warpedImageRects[i] = cv::Rect2i(imageMinmaxX.first->x, imageMinmaxY.first->y, imageColNum, imageRowNum);
        _biasedHomographies[i] = _scaledHomographies[i].clone();
        _biasedHomographies[i].row(0) += -1.0 * imageMinmaxX.first->x * _biasedHomographies[i].row(2);
        _biasedHomographies[i].row(1) += -1.0 * imageMinmaxY.first->y * _biasedHomographies[i].row(2);
    }
}


void MosaicCreator::warpNonBlended(int i)
{
    std::mutex mosaicMutex;

    cv::Mat image = cv::imread(_imageAligner._filePathList[i]);	
    cv::cvtColor(image, image, COLOR_BGR2BGRA);

    cv::Mat warpedImage = cv::Mat(_warpedImageRects[i].height, _warpedImageRects[i].width, CV_8UC4, Scalar(0, 0, 0, 0));
    cv::warpPerspective(image, warpedImage, _biasedHomographies[i], warpedImage.size(), cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_TRANSPARENT);

    cv::Rect copyToRects{_xMosaicBias + _warpedImageRects[i].x, _yMosaicBias + _warpedImageRects[i].y, warpedImage.cols, warpedImage.rows};
    cv::Mat warpedImageAlpha;
    cv::extractChannel(warpedImage, warpedImageAlpha, 3);

    mosaicMutex.lock();
    warpedImage.copyTo(_mosaicImage(copyToRects), warpedImageAlpha > 0);
    mosaicMutex.unlock();

	_distancesFromCenter[i] = 0.5 * cv::norm(cv::Point2i(_mosaicImage.cols, _mosaicImage.rows) - (copyToRects.br() + copyToRects.tl()));

}


void MosaicCreator::warpBlended(int i)
{
    std::mutex mosaicMutex;

    cv::Mat borderedImage, image = cv::imread(_imageAligner._filePathList[i]);	
	copyMakeBorder(image, borderedImage, _borderSize, _borderSize, _borderSize, _borderSize, cv::BORDER_REPLICATE);

	cv::Mat mosaicImageTemp = cv::Mat(_mosaicRowNum + 2 * _borderSize / _resizedFactorForMosaic, 
	                                  _mosaicColNum + 2 * _borderSize / _resizedFactorForMosaic, CV_8UC3, Scalar(0, 0, 0));

    cv::Mat warpedImage = cv::Mat(_warpedImageRects[i].height + 2 * _borderSize / _resizedFactorForMosaic, 
	                              _warpedImageRects[i].width  + 2 * _borderSize / _resizedFactorForMosaic, CV_8UC3, Scalar(0, 0, 0));

    cv::warpPerspective(borderedImage, warpedImage, _biasedHomographies[i], warpedImage.size(), cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT);

	cv::Rect copyToRects{_xMosaicBias + _warpedImageRects[i].x, _yMosaicBias + _warpedImageRects[i].y, warpedImage.cols, warpedImage.rows};
 	warpedImage.copyTo(mosaicImageTemp(copyToRects));

	std::vector<cv::Point2d> originalCorners{cv::Point2d{_borderSize, _borderSize}, cv::Point2d{_borderSize + image.cols, _borderSize}, 
	                                         cv::Point2d{_borderSize + image.cols, _borderSize + image.rows}, cv::Point2d{_borderSize, _borderSize + image.rows}};
	cv::perspectiveTransform(originalCorners, originalCorners, _biasedHomographies[i]);

	std::vector<cv::Point2i> warpedCorners(originalCorners.begin(), originalCorners.end());
	cv::Mat mosaicImageMask = cv::Mat(mosaicImageTemp.rows, mosaicImageTemp.cols, CV_8UC1, Scalar(0));

	for (auto & p : warpedCorners) {
		p = p + cv::Point2i{_xMosaicBias + _warpedImageRects[i].x, _yMosaicBias + _warpedImageRects[i].y};
	}
	cv::fillConvexPoly(mosaicImageMask, warpedCorners, Scalar(255), 4);     

	mosaicMutex.lock();
	cv::Mat newMask;
	cv::bitwise_and(_blendedMask, mosaicImageMask, newMask);
	_blender.feed(mosaicImageTemp, (mosaicImageMask - newMask), Point2f(0, 0));
	cv::bitwise_or(_blendedMask, mosaicImageMask, _blendedMask);
	mosaicMutex.unlock();

	_distancesFromCenter[i] = 0.5 * cv::norm(cv::Point2i(mosaicImageTemp.rows, mosaicImageTemp.cols) - (copyToRects.br() + copyToRects.tl()));

}


void MosaicCreator::saveMosaicImage(std::string outputPath)
{

	std::vector<int> imageRange(_imageAligner._imgNum);
	std::iota(imageRange.begin(), imageRange.end(), 0);

	float percent = 0.0;
	
	

    std::mutex mosaicMutex;

	std::for_each(std::execution::seq, imageRange.begin(), imageRange.end(), [&](auto i) {

		if (cv::determinant(_imageAligner._alignModelList[_imageAligner.findVisitIndex(i)]) < 0.3) {
			percent += 1.0f / (float)_imageAligner._imgNum;
			Utils::printProgress(percent); 
			return;
		}

        if (_blendImages)
        {
		    warpBlended(i);
        }
        else
        {
			warpNonBlended(i);
        }

		percent += 1.0f / (float)_imageAligner._imgNum;
		Utils::printProgress(percent); 
	});

	int centerImageIndex = std::min_element(_distancesFromCenter.begin(), _distancesFromCenter.end()) - _distancesFromCenter.begin();	
	std::cout << "center " << _imageAligner._filePathList[centerImageIndex] << std::endl;

	if (_blendImages) {
		cv::Mat blendedMask, blendedImage;
		_blender.blend(blendedImage, blendedMask);
		blendedImage.convertTo(blendedImage, CV_8UC3);
		cv::cvtColor(blendedImage, blendedImage, COLOR_BGR2BGRA);
		std::vector<cv::Mat>channels(4);
		cv::split(blendedImage, channels);
		channels[3] = blendedMask;
		cv::merge(channels, _mosaicImage);
	}

	cv::imwrite(outputPath, _mosaicImage);
	std::cout << std::endl;
}