#include <iostream>
// #include <execution>
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
#include <tbb/parallel_for.h>

struct SticherArgs : public argparse::Args {
    std::string &image_directory   = arg("image_directory", "path to input images to stitch");
    std::string &output_directory  = arg("output_directory", "path to input images to stitch");
    int &image_num_limit           = kwarg("N,n", "work on only the first N images").set_default(std::numeric_limits<int>::max());
	bool &images_ordered           = kwarg("ordered", "are images ordered by name").set_default(true);
	int &resized_image_width       = kwarg("image_width", "width of resized images").set_default(1500);
};

extern std::string Utils::baseDir;

int main(int argc, char* argv[])
{
	try {

		auto start_time = std::chrono::system_clock::now();

		SticherArgs args = argparse::parse<SticherArgs>(argc, argv);

		Utils::baseDir = args.output_directory;
		std::filesystem::create_directory(Utils::baseDir);
		std::filesystem::create_directory(Utils::baseDir + "/cache/");
		std::filesystem::create_directory(Utils::baseDir + "/cache/matchPtfile");
		std::filesystem::create_directory(Utils::baseDir + "/cache/keyPtfile");

		std::set<std::string> allowed_extensions({".JPG", ".JPEG", ".PNG", ".TIFF", ".jpg", ".jpeg", ".png", ".tiff"});
		std::set<std::string> imagePathSet, resizedImagePathSet;

		for (auto &p : std::filesystem::directory_iterator(args.image_directory))
		{
			if (allowed_extensions.find(p.path().extension()) != allowed_extensions.end()) {
				imagePathSet.insert(p.path());
			}
		}

		std::filesystem::create_directories(Utils::baseDir + "/resized_images/");

		for(auto imgPath : imagePathSet) {
			cv::Mat image = imread(imgPath);
			cv::resize(image, image, cv::Size(args.resized_image_width, int(image.rows * args.resized_image_width / image.cols)));
			std::string resized_path = Utils::baseDir + "/resized_images/" + std::filesystem::path(imgPath).filename().string();
			cv::imwrite(resized_path, image);
			image.release(); // free mem
			resizedImagePathSet.insert(resized_path);
			std::cout << "Added image " << resized_path << '\n';
			if (resizedImagePathSet.size() > args.image_num_limit) 
				break;
		};

		auto end = std::next(resizedImagePathSet.begin(), std::min(args.image_num_limit, (int)resizedImagePathSet.size()));
		std::vector<std::string> imagePathList(resizedImagePathSet.begin(), end);

		// for (auto imgPath: originalImagePathList) {
		// 	cv::Mat image = imread(imgPath);
		// 	cv::resize(image, image, cv::Size(args.resized_image_width, int(image.rows * args.resized_image_width / image.cols)));
		// 	std::string resized_path = Utils::baseDir + "/resized_images/" + std::filesystem::path(imgPath).filename().string();
		// 	cv::imwrite(resized_path, image);
		// 	imagePathList.push_back(resized_path);
    	// 	std::cout << "Added image " << resized_path << '\n';
		// }

		std::vector<cv::Point2d> PtSet1, PtSet2;
		auto matcher = new PointMatcher(imagePathList);

		std::vector<std::pair<int, int>> clusters;
		int cluster_start = 0, cluster_end = 0;
		bool matched;
		for (int i = 0; i < imagePathList.size() - 1; i++) {
			try {
				matched = matcher->featureMatcher(i, i + 1, PtSet1,PtSet2); 
			} catch(...) {
				matched = false;
			}
			if (!matched) {
				cluster_end = i;
				clusters.push_back(std::pair<int, int> {cluster_start, cluster_end});
				cluster_start = i + 1;
			}
			if (i == imagePathList.size() - 2) {
				clusters.push_back(std::pair<int, int> {cluster_start, i});
			}
		}

		for (auto i : clusters) {
			try {
				std::cout << i.first << " " << i.second << std::endl; 
				if (i.first >= i.second - 1) {
					continue;
				}
				std::vector<std::string> clusterPathList;
				for (int j = i.first; j < i.second - 1; j++) {
					clusterPathList.push_back(imagePathList[j]);
				} 

				ImageAligner imgAligner(clusterPathList);
				int referNo = 1;
				Utils::baseDir = args.output_directory + std::to_string(i.first);
				std::filesystem::create_directory(Utils::baseDir);
				std::filesystem::create_directory(Utils::baseDir + "/cache/");
				std::filesystem::create_directory(Utils::baseDir + "/cache/matchPtfile");
				std::filesystem::create_directory(Utils::baseDir + "/cache/keyPtfile");
				imgAligner.imageStitcherbyGroup(referNo);
			} catch(...) {
				continue;
			}
		}
		

		auto end_time = std::chrono::system_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
		std::cout << "Stitching took " << elapsed.count() << " seconds\n";

	} catch (const std::exception& e) {
		
        std::cout << "Caught exception " << e.what() << ", exiting." << std::endl;
		return 1;
	}
	
	return 0;
}
