// config.h
#pragma once
#include <filesystem>
#include <string>
#include <iostream>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

struct UsgsConfig {
    std::string username;
    std::string token;
    std::string dataset;
};

struct SearchConfig {
    double minLon = 0.0, minLat = 0.0, maxLon = 0.0, maxLat = 0.0;
    std::string startDate; // "YYYY-MM-DD"
    std::string endDate;
    int maxCloud = 100;
};

struct ResampleConfig {
    std::string crs;
    double xRes = 30.0;
    double yRes = 30.0;
    std::string method; // "bilinear", ...
};

struct PathConfig {
    fs::path rootDir;
    fs::path workdir;
    fs::path downloadDir;
    fs::path outputDir;
};

class config
{
public:
    static fs::path rootPath;
    static fs::path pythonPath;

    static UsgsConfig     usgs;
    static SearchConfig   search;
    static ResampleConfig resample;
    static PathConfig     paths;

    static int read();

    static void print() {
        std::cout << "Python Path : " << pythonPath << '\n';
        std::cout << "USGS User   : " << usgs.username << '\n';
        std::cout << "Dataset     : " << usgs.dataset << '\n';
        std::cout << "Search bbox : "
            << search.minLon << ", " << search.minLat << ", "
            << search.maxLon << ", " << search.maxLat << '\n';
        std::cout << "Resample    : " << resample.crs
            << " (" << resample.xRes << ", " << resample.yRes
            << ") " << resample.method << '\n';
    }
};
