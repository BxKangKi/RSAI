// config.cpp
#include "config.h"
#include <algorithm>
#include <fstream>


fs::path config::rootPath;
fs::path config::pythonPath;

UsgsConfig config::usgs;
SearchConfig config::search;
ResampleConfig config::resample;
PathConfig config::paths;

namespace {
    constexpr const char* kDefaultPythonPath = "";
    constexpr const char* kDefaultDataset = "landsat_ot_c2_l2";
    constexpr const char* kDefaultDate = "2023-01-01";
    constexpr const char* kDefaultCRS = "EPSG:5179";
    constexpr const int kMaxCloud = 100;
    constexpr const double kResampleRes = 30.0;
    constexpr const char* kResampleMethod = "bilinear";
    constexpr const char* kDefaultWorkDir = "C:/ERSML/Data";

    void write_default_config(const fs::path& path) {
        json def;

        def["pythonPath"] = kDefaultPythonPath;

        def["usgs"] = {
            {"username", ""},
            {"token",    ""},
            {"dataset",  kDefaultDataset}
        };

        def["search"] = {
            {"bbox",        {126.5, 37.2, 127.2, 37.8}},
            {"start_date",  kDefaultDate},
            {"end_date",    kDefaultDate},
            {"max_cloud",   kMaxCloud}
        };

        def["resample"] = {
            {"crs",     kDefaultCRS},
            {"x_res",   kResampleRes},
            {"y_res",   kResampleRes},
            {"method",  kResampleMethod}
        };

        def["paths"] = {
            {"workdir",      kDefaultWorkDir},
            {"download_dir", "/download"},
        };

        std::ofstream ofs(path);
        if (!ofs.is_open()) {
            std::cerr << "Failed to create config.json\n";
            return;
        }
        ofs << def.dump(4);
    }

    int make_default_config() {
        config::pythonPath = kDefaultPythonPath;

        config::usgs.username = "";
        config::usgs.token = "";
        config::usgs.dataset = kDefaultDataset;

        config::search.minLon = 126.5;
        config::search.minLat = 37.2;
        config::search.maxLon = 127.2;
        config::search.maxLat = 37.8;
        config::search.startDate = kDefaultDate;
        config::search.endDate = kDefaultDate;
        config::search.maxCloud = kMaxCloud;

        config::resample.crs = kDefaultCRS;
        config::resample.xRes = kResampleRes;
        config::resample.yRes = kResampleRes;
        config::resample.method = kResampleMethod;

        config::paths.workdir = kDefaultWorkDir;
        config::paths.downloadDir = "/download";
        config::paths.outputDir = "/output";

        config::print();
        return 0;
    }
}

int config::read()
{
    const fs::path path{ "config.json" };

    // 1) 파일이 없으면 기본 설정 쓰고 그걸 반환
    if (!fs::exists(path)) {
        write_default_config(path);
        return make_default_config();
    }

    // 2) 파일이 있는데 열기 실패 → 기본값
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config.json\n";
        return make_default_config();
    }

    // 3) JSON 파싱
    json data;
    try {
        file >> data;
    }
    catch (const json::parse_error& e) {
        std::cerr << "JSON parse error in config.json: " << e.what() << '\n';
        // 손상된 경우 기본 설정으로 덮어쓰고 기본값 반환
        write_default_config(path);
        return make_default_config();
    }

    // 4) 정상 케이스

    config::pythonPath = data.value("pythonPath", std::string{ kDefaultPythonPath });

    // usgs
    if (data.contains("usgs") && data["usgs"].is_object()) {
        const auto& u = data["usgs"];
        config::usgs.username = u.value("username", std::string{});
        config::usgs.token = u.value("token", std::string{});
        config::usgs.dataset = u.value("dataset", std::string{ kDefaultDataset });
    }

    // search
    if (data.contains("search") && data["search"].is_object()) {
        const auto& s = data["search"];
        if (s.contains("bbox") && s["bbox"].is_array() && s["bbox"].size() == 4) {
            config::search.minLon = s["bbox"][0].get<double>();
            config::search.minLat = s["bbox"][1].get<double>();
            config::search.maxLon = s["bbox"][2].get<double>();
            config::search.maxLat = s["bbox"][3].get<double>();
        }
        config::search.startDate = s.value("start_date", std::string{});
        config::search.endDate = s.value("end_date", std::string{});
        config::search.maxCloud = s.value("max_cloud", 100);
    }

    // resample
    if (data.contains("resample") && data["resample"].is_object()) {
        const auto& r = data["resample"];
        config::resample.crs = r.value("crs", std::string{});
        config::resample.xRes = r.value("x_res", 30.0);
        config::resample.yRes = r.value("y_res", 30.0);
        config::resample.method = r.value("method", std::string{ "bilinear" });
    }

    // paths
    if (data.contains("paths") && data["paths"].is_object()) {
        const auto& p = data["paths"];
        config::paths.workdir = p.value("workdir", std::string{});
        config::paths.downloadDir = p.value("download_dir", std::string{});
        config::paths.outputDir = p.value("output_dir", std::string{});
    }

    config::print();
    return 0;
}
