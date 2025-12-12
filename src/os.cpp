#include "os.h"
#include "config.h"
#include "helper.h"

void os::start()
{
#if WINDOWS
	windows::start();
#endif
}


#if WINDOWS
#include <windows.h>

std::wstring windows::to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size);
    return wstr;
}

bool windows::cd(const std::string& dir)
{
    std::string cmd;
    cmd = "cd /d";
    addPath(cmd, dir);
#if _DEBUG
    std::cout << cmd << '\n';
#endif
    std::system(cmd.c_str());
    return true;
}

std::string windows::getCurrentDir() {
    char buf[MAX_PATH];
    DWORD len = GetCurrentDirectoryA(MAX_PATH, buf);
    return std::string(buf, len);
}

void windows::start()
{
    fs::path root = toUnixPath(getCurrentDir());
    config::paths.rootDir = root;
    std::cout << "Current Root is: " << root.string() << "\n";

    _putenv_s("Path", root.string().c_str());

    // GDAL_DATA와 PROJ_DATA 설정
    std::filesystem::path gdal = root / "bin" / "GDAL";
    std::filesystem::path gdalApps = root / "bin" / "GDAL" / "gdal" / "apps";
    std::filesystem::path python = root / "bin" / "Python";
    std::filesystem::path gdalData = root / "bin" / "GDAL" / "gdal-data";
    std::filesystem::path projData = root / "bin" / "GDAL" / "proj9" / "share";

    std::string newPath = toUnixPath(gdal).c_str();
    newPath += ";";
    newPath += toUnixPath(gdalApps).c_str();
    newPath += ";";
    newPath += toUnixPath(python).c_str();
    // PATH 설정
    _putenv_s("Path", newPath.c_str());
    _putenv_s("GDAL_DATA", toUnixPath(gdalData).c_str());
    _putenv_s("PROJ_LIB", toUnixPath(projData).c_str());

# if _DEBUG
    std::cout << "Path: " << newPath << '\n';
    std::cout << "GDAL_DATA: " << toUnixPath(gdalData) << '\n';
    std::cout << "PROJ_DATA: " << toUnixPath(projData) << '\n';
#endif
}

#endif