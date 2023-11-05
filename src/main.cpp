#include <iostream>
#include <filesystem>
#include <chrono>
#include <set>

#include "argparse.hpp"
#include "featureMatch.h"
#include "alignment.h"
#include "util.h"

struct SticherArgs : public argparse::Args {
    std::string &image_directory   = arg("image_directory", "path to input images to stitch");
    std::string &output_directory  = arg("output_directory", "path to input images to stitch");
    int &image_num_limit           = kwarg("N,n", "work on only the first N images - 0 will use all").set_default(0);
	bool &images_ordered           = kwarg("ordered", "are images ordered by name").set_default(true);
};

int main(int argc, char* argv[])
{
	try {

		auto start_time = std::chrono::system_clock::now();

		SticherArgs args = argparse::parse<SticherArgs>(argc, argv);

		std::set<std::string> allowed_extensions({".JPG", ".JPEG", ".PNG", ".TIFF", ".jpg", ".jpeg", ".png", ".tiff"});
		std::set<std::string> imagePathList;

		for (auto &p : std::filesystem::directory_iterator(args.image_directory))
		{
			if (allowed_extensions.find(p.path().extension()) != allowed_extensions.end()) {
				imagePathList.insert(p.path());
			}
		}

		ImageAligner imgAligner(std::vector<std::string>(imagePathList.begin(), imagePathList.end()));
		int referNo = 1;

		imgAligner.imageStitcherbyGroup(referNo);

		auto end_time = std::chrono::system_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
		std::cout << "Stitching took " << elapsed.count() << " seconds\n";

	} catch (const std::exception& e) {
		
        std::cout << "Caught exception " << e.what() << ", exiting." << std::endl;
		return 1;
	}
	
	return 0;
}
