#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace {

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

extern "C" bool firewing_torch_topk_bf16(
    const std::uint16_t* values,
    std::size_t count,
    std::size_t k,
    std::size_t* indices) {
    if (values == nullptr || indices == nullptr || k == 0 || k > count) {
        return false;
    }
    using entry = std::pair<float, std::int64_t>;
    std::vector<entry> queue(count);
    for (std::size_t index = 0; index < count; ++index) {
        queue[index] = {bf16_to_float(values[index]), static_cast<std::int64_t>(index)};
    }
    const auto greater = [](const entry& left, const entry& right) {
        return left.first > right.first;
    };
    if (k * 64 <= count) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), greater);
    } else {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), greater);
        std::sort(queue.begin(), queue.begin() + k - 1, greater);
    }
    for (std::size_t index = 0; index < k; ++index) {
        indices[index] = static_cast<std::size_t>(queue[index].second);
    }
    return true;
}
