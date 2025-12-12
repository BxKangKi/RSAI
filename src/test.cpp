#include "test.h"
#include "reproject.h"


namespace test {

    int test_run()
    {
        fs::path in = config::paths.workdir.string() + "\\input\\input.tif";
        fs::path out = config::paths.workdir.string() + "\\output\\output.tif";
        gdal::reproject().execute(in, out);
        return 0;
    }
}
