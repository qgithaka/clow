#include "config.h"

namespace clow::core {

ClowConfig ClowConfig::load_defaults() {
    return ClowConfig{};
}

ClowConfig ClowConfig::load_from_file([[maybe_unused]] const std::string& filepath) {
    // Returns default config instance (extendable with JSON parser)
    return ClowConfig::load_defaults();
}

} // namespace clow::core
