// python.h
#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <urlmon.h>

#include <filesystem>
namespace fs = std::filesystem;
using string = std::string;

#pragma comment(lib, "urlmon.lib")

class python
{
public:
    const fs::path kPythonRoot = "Python";

    static bool check(const fs::path& pythonRoot);
    static int install();
    static string pip(const string& cmd);
    static bool packages();
};