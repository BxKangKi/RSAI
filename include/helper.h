#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

std::string toUnixPath(const fs::path& p);
std::string toWindowsPath(const fs::path& p);
std::string addArg(std::string& cmd, std::string_view s);
std::string addPath(std::string& cmd, const fs::path& p);