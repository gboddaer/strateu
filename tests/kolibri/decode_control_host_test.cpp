// Host-only publication regression: must compile as C++17, just like nvcc's .cu files.
#include "strata/kolibri/decode_wait.hpp"
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <thread>

int main() {
    using namespace strata::kolibri;
    DecodeWaitControl control{};
    for (uint32_t value : {0u, 1u, 0x80000000u, 0xffffffffu}) {
        decode_control_store(control.ready, value);
        assert(decode_control_load(control.ready) == value);
    }
    decode_control_store(control.ready, 0);
    uint32_t payload[16]{};
    constexpr uint32_t rounds = 20000;
    std::thread producer([&] {
        for (uint32_t seq = 1; seq <= rounds; ++seq) {
            while (decode_control_load(control.acknowledged) != seq - 1) std::this_thread::yield();
            for (uint32_t i = 0; i < 16; ++i) payload[i] = seq * 17u + i;
            decode_control_store(control.ready, seq);
        }
    });
    for (uint32_t seq = 1; seq <= rounds; ++seq) {
        while (decode_control_load(control.ready) != seq) std::this_thread::yield();
        for (uint32_t i = 0; i < 16; ++i) assert(payload[i] == seq * 17u + i);
        decode_control_store(control.acknowledged, seq);
    }
    producer.join();
}
