#include "system.h"
#include <iostream>
#include <string>
#include "python.h"
#include "os.h"
#include "helper.h"

#define PYTHON_PATH "/bin/Python/";

int system::start()
{
    // start from os class
    os::start();
    config::read();
    if (startPython() != 0)
    {
        return 1;
    }
    return 0;
}


int system::startPython()
{
    // load python
    if (python::install() != 0) {
        return 1;
    }
    return 0;
}