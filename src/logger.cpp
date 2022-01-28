#include "../include/logger.h"
//#include <filesystem>
#include <experimental/filesystem>
#include <iomanip>

namespace logger {
    Logger logger;

    Logger::Logger() = default;

    void Logger::init(const std::string &log_dir, const std::string &sparse_file, int mpi_rank) {
        _mpi_rank = mpi_rank;
        _log_prefix = "[" + std::to_string(_mpi_rank) + "] ";

        std::experimental::filesystem::create_directory(log_dir);
        std::string log_file = log_dir + "rank_" + std::to_string(_mpi_rank) + "_" + sparse_file.substr(sparse_file.length() - 16);
        this->ofs.open(log_file, std::ofstream::out);
        this->ofs_timer.open(log_dir + "rank_" + std::to_string(_mpi_rank) + "_timer" + "_" + sparse_file.substr(sparse_file.length() - 16),
                             std::ofstream::out);
        this->ofs_timer.precision(5);
        *this << "Logger initialization...\n";
    }

    void Logger::shutdown() {
        this->ofs.close();
    }

    void Logger::new_stage(const std::string &stage) {
        *this << "\n\n" + std::string(35, '#') + " "
                 + stage + " " + std::string(35, '#') << "\n" << std::endl;
    }

    void Logger::func_entry(const std::string &string) {
        *this << "Entry [" << string << "]" << std::endl;
    }

    void Logger::func_exit(const std::string &string) {
        *this << "Exit [" << string << "]" << std::endl;
    }

    void Logger::start(timer::Timer &timer, const int line, const std::string &desc) {
        ofs_timer << "\nstart: line: " << line << ": " << desc << std::endl;
        ofs_timer << std::flush;
        timer.start();
    }

    double Logger::end(timer::Timer &timer, const int line, const std::string &desc) {
        double duration = timer.end();
        ofs_timer << "end: line: " << line << ": " << desc << ", duration: " << std::fixed << std::setprecision(7) << duration << std::endl;
        ofs_timer << std::flush;
        return duration;
    }
}