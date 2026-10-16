// include/strata/kolibri/engine.hpp - the Kolibri 1 (GGUF architecture kolibri1) engine path as a library:
// `strata` hands a kolibri1 model to it at the top of main() (the family switch), and strata-kolibri is a thin
// wrapper around the same entry point.  See src/kolibri/kolibri_engine.cu.
#pragma once

namespace strata::kolibri {

/// The engine's command line (strata's flag names where they mean the same thing: --native / --model,
/// --max-context, --prefill / --chunk, --serve, --tokens, --max-new, --no-token-graph; the family's own:
/// --expert-vram, --expert-slots, --cpu-threads).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::kolibri
