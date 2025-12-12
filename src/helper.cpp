#include "helper.h"

std::string toUnixPath(const fs::path& p)
{
    std::string s = p.string();
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
}

std::string toWindowsPath(const fs::path& p)
{
    std::string s = p.string();
    std::replace(s.begin(), s.end(), '/', '\\');
    return s;
}

std::string addArg(std::string& cmd, std::string_view s)
{
    cmd += ' ';
    cmd += s;
    return cmd;
}

std::string addPath(std::string& cmd, const fs::path& p)
{
    cmd += " \"";
    cmd += toUnixPath(p);
    cmd += '"';
    return cmd;
}