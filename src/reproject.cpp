#include "reproject.h"
#include "helper.h"
#include <iostream>

namespace gdal
{
    bool reproject::execute(
        const fs::path& inFile,
        const fs::path& outFile)
    {
        std::string cmd = "gdal raster reproject";
        addArg(cmd, "--overwrite");
        addArg(cmd, "-i");
        addPath(cmd, inFile);
        addArg(cmd, "-o");
        addPath(cmd, outFile);

        std::cout << cmd << '\n';
        std::system(cmd.c_str());

        return true;
    }
}
