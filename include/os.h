#pragma once

#include <iostream>
#include <string>
#include <filesystem>
#include "config.h"

namespace fs = std::filesystem;

class os
{
public:
	static void start();
};

#if WINDOWS

class windows
{
public:
	static std::wstring to_wstring(const std::string &str);
	static std::string getCurrentDir();
	static bool cd(const std::string &dir);
	static void start();
};
#endif