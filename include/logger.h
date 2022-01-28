#ifndef MPI_MATRIX_MULTIPLICATION_LOGGER_H
#define MPI_MATRIX_MULTIPLICATION_LOGGER_H

#include <iostream>
#include <fstream>
#include <chrono>

namespace timer {
    using std::cout; using std::endl;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    using std::chrono::milliseconds;
    using std::chrono::seconds;
    using std::chrono::system_clock;

    class Timer {
        double _start;
        double _end;

    public:


        void start() {
            _start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        }

        double end() {
            _end = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
            return _end - _start;
        }
    };

}

/**
 * https://levelup.gitconnected.com/logging-in-c-60cd1571df15
 */
namespace logger {
    struct Logger {
        std::ofstream ofs;
        int _mpi_rank;
        std::string _log_prefix;
        bool was_new_line = true;
        bool on = true;
        std::ofstream ofs_timer;

        Logger();

        void init(const std::string &log_dir, const std::string &sparse_file, int mpi_rank);

        void shutdown();

        void new_stage(const std::string &stage);

        template<typename T>
        Logger &operator<<(T t);

        // this is needed so << std::endl works
        Logger &operator<<(std::ostream &(*fun)(std::ostream &));

        void func_entry(const std::string &string);

        void func_exit(const std::string &string);

        void start(timer::Timer &timer, const int line, const std::string &desc);

        double end(timer::Timer &timer, const int line, const std::string &desc);
    };

    template<typename T>
    inline Logger &Logger::operator<<(T msg) {
        if (on) {
            if (was_new_line) {
                ofs << _log_prefix;
                was_new_line = false;
            }
            ofs << msg;
            ofs.flush();
        }
        return *this;
    }

    inline Logger &Logger::operator<<(std::ostream &(*fun)(std::ostream &)) {
        if (on) {
            was_new_line = true;
            ofs << "\n" << std::endl;
            ofs.flush();
        }
        return *this;

    }

    /** Default logger instance */
    extern Logger logger;
}

#endif //MPI_MATRIX_MULTIPLICATION_LOGGER_H