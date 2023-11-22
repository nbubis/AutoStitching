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

#include <tbb/parallel_for.h>

struct SticherArgs : public argparse::Args {
    std::string &imageDirectory     = arg("image_directory", "path to input images to stitch");
    std::string &outputDirectory    = arg("output_directory", "path to input images to stitch");
    unsigned int &imageNumLimit     = kwarg("N,n", "work on only the first N images").set_default(std::numeric_limits<int>::max());
	bool &imagesOrdered             = kwarg("ordered", "are images ordered by name").set_default(true);
	unsigned int &resizedImageWidth = kwarg("image_width", "width of resized images").set_default(1500);
};

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

	std::vector<uchar> status;
	cv::Mat Fmatrix = cv::findFundamentalMat(initialMatchedPoints1, initialMatchedPoints2, cv::RANSAC, 1.5, 0.99, status);

	if (Fmatrix.empty()) {
		cout << "Empty fundamental matrix" << endl;
		return false;
	}
	std::vector<cv::Point2d> FundamentalMatchedPoints1, FundamentalMatchedPoints2;

	for (int i = 0; i < status.size(); i++)
	{
		if (status[i] == 1)
		{
			FundamentalMatchedPoints1.push_back(initialMatchedPoints1[i]);
			FundamentalMatchedPoints2.push_back(initialMatchedPoints1[i]);
		}
	}
	if (FundamentalMatchedPoints1.size() < 30)
	{
		return false;
	}

	cv::Mat homography = cv::findHomography(FundamentalMatchedPoints1, FundamentalMatchedPoints2, cv::RANSAC, 2.5);

	std::vector<cv::Point2d> HomographyMatchedPoints1, HomographyMatchedPoints2, warpedPoints;

	if (homography.empty()) {
		return false;
	}
	cv::perspectiveTransform(FundamentalMatchedPoints1, warpedPoints, homography);

	for (int i = 0; i < FundamentalMatchedPoints1.size(); i++)
	{
		if (cv::norm(warpedPoints[i] - FundamentalMatchedPoints2[i]) < 3.0)
		{
			HomographyMatchedPoints1.push_back(FundamentalMatchedPoints1[i]);
			HomographyMatchedPoints2.push_back(FundamentalMatchedPoints2[i]);
		}
	}

	if (HomographyMatchedPoints1.size() < 30)
	{
		return false;
	}
	std::cout << "Found " << HomographyMatchedPoints1.size() << " matches between " << imageData1.imagePath << " and " << imageData2.imagePath << std::endl;

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
	imagePaths.resize(std::min(args.imageNumLimit, (unsigned int)imagePaths.size()));

	// read files and extract keypoints

	Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(500);

	std::mutex mtx;
  
	std::for_each(std::execution::par_unseq, imagePaths.begin(), imagePaths.end(), [&](auto imgPath) {
			
		// read images into buffer, and only then decode to allow parralization 
		std::cout << "Reading image '" << imgPath << "'" << std::endl;
		cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
	
		if (image.empty()) {
			return;
		}

		cv::resize(image, image, cv::Size(args.resizedImageWidth, int(image.rows * args.resizedImageWidth / image.cols)));
		std::filesystem::path resized_images_directory = std::filesystem::path(args.outputDirectory) / "resized_images";
		std::filesystem::create_directories(resized_images_directory);
		std::string resized_image_path = resized_images_directory / std::filesystem::path(imgPath).filename();
		
		cv::imwrite(resized_image_path, image);
		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptors;

		detector->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);
		mtx.lock();
		imageStitchData.push_back({resized_image_path, keyPoints, descriptors, args.resizedImageWidth});
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

	std::cout << "Reading Images ..." << std::endl;
	std::vector<ImageStitchData> imageStitchData = readImageData(args);
	std::cout << "Read images and extracted keypoints." << std::endl;

	int cluster_start = 0, cluster_end = 0;

	std::vector<std::vector<ImageStitchData>> clustersData;

	for (size_t i = 0; i < imageStitchData.size() - 1; ++i) {
		bool matched = matchFeatures(imageStitchData[i], imageStitchData[i + 1]);
		if (!matched || (i - cluster_start + 1) >= 150 || i == (imageStitchData.size() - 2)) {
			cluster_end = i;
			if (cluster_end - cluster_start >= 2) {
				std::vector<ImageStitchData> clusterData(imageStitchData.begin() + cluster_start, imageStitchData.begin() + cluster_end + 1);
				clustersData.push_back(clusterData);
				std::cout << "Created cluster [" << cluster_start << ", " << cluster_end << "] " << std::endl; 
			} 
			cluster_start = i + 1;
		}
	}
	
	std::for_each(std::execution::par_unseq, clustersData.begin(), clustersData.end(), [&](auto clusterData) {
	// std::for_each(clustersData.begin(), clustersData.end(), [&](auto clusterData) {

		std::vector<std::string> imagePathsList;
		std::transform(clusterData.begin(), clusterData.end(), std::back_inserter(imagePathsList), 
			[&] (auto imageStitchData) {return imageStitchData.imagePath;});

		std::filesystem::path outputDirectory = std::filesystem::path(args.outputDirectory) / 
			(std::filesystem::path(imagePathsList.front()).filename().string() + std::string("_") +
			 std::filesystem::path(imagePathsList.back()).filename().string());
		std::filesystem::create_directories(outputDirectory / "cache/keyPtfile");
		std::filesystem::create_directories(outputDirectory / "cache/matchPtfile");	
		std::cout << "Stitching " << outputDirectory << std::endl;
		try {	
			ImageAligner imgAligner(imagePathsList, outputDirectory);
			imgAligner.imageStitcherbyGroup(-1);
		} catch (...) {
			std::cout << "Failed to stitch " << outputDirectory << ", continuing" << std::endl;
			return;
		}

    });
		
	auto end_time = std::chrono::system_clock::now();

	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
	std::cout << "Stitching took " << elapsed.count() << " seconds\n";

	
	return 0;
}
