#include <iostream>
#include <execution>
#include <filesystem>
#include <limits>
#include <utility>
#include <chrono>
#include <set>

#include "topology.h"
#include "argparse.hpp"
#include "featureMatch.h"
#include "alignment.h"
#include "util.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "tbb/tbb.h"

struct SticherArgs : public argparse::Args {
    std::string &imageDirectory    = arg("image_directory", "path to input images to stitch");
    std::string &outputDirectory    = arg("output_directory", "path to input images to stitch");
    unsigned int &imageNumLimit     = kwarg("N,n", "work on only the first N images").set_default(std::numeric_limits<int>::max());
	bool &imagesOrdered             = kwarg("ordered", "are images ordered by name").set_default(true);
	unsigned int &resizedImageWidth = kwarg("image_width", "width of resized images").set_default(1500);
};

extern std::string Utils::baseDir;

struct ImageStitchData {
	std::string imagePath;
	std::vector<cv::KeyPoint> keyPoints;
	cv::Mat descriptors;
	unsigned int resizedImageWidth;
};


bool matchFeatures(ImageStitchData & imageData1, ImageStitchData & imageData2) {

	cv::FlannBasedMatcher matcher; 
	std::vector<std::vector<cv::DMatch>> knnMatches;

	int num1 = imageData1.keyPoints.size(), num2 = imageData2.keyPoints.size();

	if (num1 < 5 || num2 < 5) {
		return false;
	}

	matcher.knnMatch(imageData1.descriptors, imageData2.descriptors, knnMatches, 5);   
	// matches – Matches. Each matches[i] is k or less matches for the same query descriptor.
	// The matches are returned in the distance increasing order

	std::vector<cv::DMatch> initialMatches;

    double minimalDistance = (*std::min_element(knnMatches.begin(), knnMatches.end(),
         [] (auto const& lhs, auto const& rhs) {return lhs[0].distance < rhs[0].distance;}))[0].distance;

	for (int i = 0; i < knnMatches.size(); i++)
	{  
		double distanceRatio = knnMatches[i][0].distance / knnMatches[i][1].distance;
		if (knnMatches[i][0].distance < minimalDistance * 5.0 && distanceRatio < 0.7)
		{
			initialMatches.push_back(knnMatches[i][0]);
		}
	}
	
	if (initialMatches.size() < 10) {
		return false;
	}

	std::vector<cv::Point2d> initialMatchedPoints1, initialMatchedPoints2;

	for (int i = 0; i < initialMatches.size(); i ++)  
	{
		initialMatchedPoints1.push_back(imageData1.keyPoints[initialMatches[i].queryIdx].pt);
		initialMatchedPoints2.push_back(imageData2.keyPoints[initialMatches[i].trainIdx].pt);
	}

	// std::vector<uchar> status;
	// cv::Mat Fmatrix = cv::findFundamentalMat(initialMatchedPoints1, initialMatchedPoints2, cv::RANSAC, 1.5, 0.99, status);

	// for (int i = 0; i < status.size(); i++)
	// {
	// 	if (status[i] != 1)
	// 	{
	// 		initialMatchedPoints1.erase(initialMatchedPoints1.begin() + i);
	// 		initialMatchedPoints2.erase(initialMatchedPoints2.begin() + i);
	// 	}
	// }
	// if (initialMatchedPoints1.size() < 10)
	// {
	// 	return false;
	// }

	cv::Mat homography = cv::findHomography(initialMatchedPoints1, initialMatchedPoints2, cv::RANSAC, 2.5);
	std::vector<cv::Point2d> warpedPoints;
	// std::cout << initialMatchedPoints1.size() << " " << homography << std::endl;
	if (homography.empty()) {
		return false;
	}
	cv::perspectiveTransform(initialMatchedPoints1, warpedPoints, homography);

	for (int i = 0; i < initialMatchedPoints1.size(); i++) 
	{
		double dist = cv::norm(warpedPoints[i] - initialMatchedPoints1[i]);
		if (dist > 3.0)
		{
			initialMatchedPoints1.erase(initialMatchedPoints1.begin() + i);
			initialMatchedPoints2.erase(initialMatchedPoints2.begin() + i);
		}
	}

	if (initialMatchedPoints1.size() < 10)
	{
		return false;
	}

	// saveMatchPts(imgIndex1, imgIndex2, pointSet1, pointSet2);
//	drawMatches(imgIndex1, imgIndex2, pointSet1, pointSet2);
	return true;
}


std::vector<ImageStitchData> readImageData(SticherArgs args) {

	std::vector<ImageStitchData> imageStitchData;

	// find relevant images
	std::set<std::string> allowed_extensions({".JPG", ".JPEG", ".PNG", ".TIFF", ".jpg", ".jpeg", ".png", ".tiff", ".jp2"});
	std::vector<std::string> imagePaths;

	for (auto &p : std::filesystem::directory_iterator(args.imageDirectory))
	{
		if (allowed_extensions.find(p.path().extension()) != allowed_extensions.end()) {
			imagePaths.push_back(p.path());
		}
	}

	std::sort(imagePaths.begin(), imagePaths.end());
	imagePaths.resize(args.imageNumLimit);

	// read files and extract keypoints

	Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(500);

	std::mutex mtx;
  
	std::for_each(std::execution::par_unseq, imagePaths.begin(), imagePaths.end(), [&](auto imgPath) {
			
		// read images into buffer, and only then decode to allow parralization 

		cv::Mat image = imread(imgPath, cv::IMREAD_COLOR);
	
		if (image.rows == 0 || image.cols == 0) {
			return;
		}

		cv::resize(image, image, cv::Size(args.resizedImageWidth, int(image.rows * args.resizedImageWidth / image.cols)));

		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptors;

		detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);
		mtx.lock();
		imageStitchData.push_back({imgPath, keyPoints, descriptors, args.resizedImageWidth});
		mtx.unlock();
		image.release();

    });

    std::sort(imageStitchData.begin(), imageStitchData.end(), [](auto & a, auto & b) { return a.imagePath < b.imagePath; });
	return imageStitchData;
}

// std::vector<std::vector<ImageStitchData>> createClusters(std::vector<ImageStitchData> imageStitchData) {
// 	return;
// }

int main(int argc, char* argv[])
{

	auto start_time = std::chrono::system_clock::now();

	SticherArgs args = argparse::parse<SticherArgs>(argc, argv);

	Utils::baseDir = args.outputDirectory;

	std::cout << "Reading Images ..." << std::endl;
	std::vector<ImageStitchData> imageStitchData = readImageData(args);
	std::cout << "Read images and extracted keypoints." << std::endl;

	int cluster_start = 0, cluster_end = 0;

	std::vector<std::vector<ImageStitchData>> clustersData;

	for (size_t i = 0; i < imageStitchData.size() - 1; ++i) {
		bool matched = matchFeatures(imageStitchData[i], imageStitchData[i + 1]);
		if (!matched) {
			cluster_end = i;
			if (cluster_end - cluster_start >= 1) {
				std::vector<ImageStitchData> clusterData(imageStitchData.begin() + cluster_start, imageStitchData.begin() + cluster_end + 1);
				clustersData.push_back(clusterData);
				std::cout << "Created cluster [" << cluster_start << ", " << cluster_end << "] " << std::endl; 
			} 
			cluster_start = i + 1;
		}
	}
	
	std::for_each(std::execution::par_unseq, clustersData.begin(), clustersData.end(), [&](auto cluster) {
		
		ImageAligner imgAligner(clusterPathList);
		imgAligner.imageStitcherbyGroup(1);

    });
		
	auto end_time = std::chrono::system_clock::now();

	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
	std::cout << "Stitching took " << elapsed.count() << " seconds\n";

	
	return 0;
}
