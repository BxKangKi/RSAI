#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <future>
#include <cstdlib>
#include <filesystem>
#include "config.h"

namespace fs = std::filesystem;
using string = std::string;

namespace gdal {
#if WIN64
	const string GDAL = "bin\\GDAL";
#else
	const std::string GDAL = "bin/gdal";
#endif
	class reproject
	{
	public:
		// using with async -> must return boolean
		bool execute(
			const fs::path& inFile,
			const fs::path& outFile);
	};
}
