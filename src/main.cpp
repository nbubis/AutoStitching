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
    std::string & imageDirectory            = arg("image_directory", "Path to input images to stitch");
    std::string & outputDirectory           = arg("output_directory", "Path to input images to stitch");
    unsigned int & imageNumLimit            = kwarg("N,n", "Work on only the first N images").set_default(std::numeric_limits<int>::max());
	unsigned int & limitImageMatchNum       = kwarg("image_match_num", "Number of images to match for each image").set_default(0);
	float        & resizedFactorForFeatures = kwarg("feature_downsample", "Factor in which to resize images for feature extraction").set_default(1.0f);
	float        & resizedFactorForMosaic   = kwarg("mosaic_downsample", "Factor in which to resize images for mosaic creation").set_default(1.0f);
};


int main(int argc, char* argv[])
{	
	// list and order relevant images
	using namespace std::chrono;
	auto start_time = system_clock::now();

	SticherArgs args = argparse::parse<SticherArgs>(argc, argv);

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

	// extract and match features
	PointMatcher pointMatcher = PointMatcher(imagePaths, args.limitImageMatchNum, args.resizedFactorForFeatures);

	// create clusters
	int cluster_start = 0, cluster_end = 0;
	std::vector<std::pair<int, int>> clusterIndices;

	for (int i = 0; i < pointMatcher.imgNum(); i++) {
		if (pointMatcher.matches()[i][i + 1].size() == 0 || (i - cluster_start + 1) >= 150 || i == pointMatcher.imgNum() - 1) {
			cluster_end = i;
			if (cluster_end - cluster_start >= 2) {
				clusterIndices.push_back({cluster_start, cluster_end});
			}
			cluster_start = i + 1;
		}
	}

	std::cout << "Created " << clusterIndices.size() << " clusters." << std::endl; 

	// align clusters
	std::for_each(std::execution::par_unseq, clusterIndices.begin(), clusterIndices.end(), [&](auto cluster) {
		PointMatcher clusterPointMatcher = pointMatcher.getSubset(cluster.first, cluster.second + 1);

		std::filesystem::path outputDirectory = 
			std::filesystem::path(args.outputDirectory) / (
			std::filesystem::path(clusterPointMatcher.imgPathList().front()).filename().string() + std::string("_") +
			std::filesystem::path(clusterPointMatcher.imgPathList().back()).filename().string());

		std::filesystem::create_directories(outputDirectory);
		ImageAligner imgAligner(clusterPointMatcher, outputDirectory);
		imgAligner.imageStitcherbyGroup(0);
		imgAligner.saveMosaicImage(args.resizedFactorForMosaic);
		imgAligner.saveMosaicImageP();
	});
		
	auto end_time = system_clock::now();
	std::cout << "Stitching took " << (duration_cast<seconds>(end_time - start_time)).count() << " seconds\n";
	
	return 0;
}
