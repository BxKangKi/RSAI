// python.cpp
#include "python.h"
#include "os.h"

bool python::check(const fs::path& pythonRoot) {
    return fs::exists(pythonRoot) && fs::is_directory(pythonRoot);
}

// env path should be registered
std::string python::pip(const std::string& cmd) {
    return "python -m pip " + cmd;
}

// env path should be registered
int python::install() {
    // pip upgrade
    std::string cmdUpgradePip = pip("install --upgrade pip");
#if _DEBUG
    std::cout << "Upgrading pip: " << cmdUpgradePip << "\n";
#endif

#if WINDOWS
    std::system(cmdUpgradePip.c_str());
#endif

    //packages();

    return 0;
}

bool python::packages()
{
    std::string command = "install -r " + config::paths.rootDir.string() + "/requirements.txt";
    std::string cmd_pip = pip(command);
#if WINDOWS
    std::system(cmd_pip.c_str());
#endif
    return true;
}