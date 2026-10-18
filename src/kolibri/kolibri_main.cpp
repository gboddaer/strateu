// src/kolibri/kolibri_main.cpp - strata-kolibri: the Kolibri 1 engine path on its own (the same entry point
// `strata` switches to for a kolibri1 model).
//
// K26: a truncated GGUF (a header that parses, a data section that does not) reaches load() and dies with an
// uncaught std::runtime_error - SIGABRT, exit 134, and a core dump.  Measured on this box with a 128 KiB
// truncation of the 47 GB Q4_K_M artifact: pre-fix exit=134 printing only "  what():  GGUF: unexpected end of
// file in header"; post-fix exit=1 printing "kolibri: load failed: GGUF: unexpected end of file in header".
// The message the engine already builds never reached the user, because the throw escaped main and
// std::terminate prints only the exception.  A CLI that crashes on a corrupt file is worse than one that
// reports it: the harnesses that drive this binary (bench.py, kl.py, the gate scripts) read the exit status,
// and 134 is indistinguishable from a genuine engine fault.  Catch at the boundary, print what the engine
// said, exit 1.
#include "strata/kolibri/engine.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    try {
        return strata::kolibri::run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kolibri: load failed: %s\n", e.what());
        return 1;
    }
}
