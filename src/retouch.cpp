#include "imglib/retouch.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace imglib {

Canvas clone_stamp(const Canvas& src,
                   int src_x,
                   int src_y,
                   int dst_x,
                   int dst_y,
                   int radius,
                   float hardness) {
  if (radius <= 0) return src;
  Canvas out = src;
  const float r_f = static_cast<float>(radius);
  const float hard = std::max(0.0f, std::min(0.99f, hardness));
  const float inner_r = r_f * hard;

  for (int dy = -radius; dy <= radius; ++dy) {
    const int ty = dst_y + dy;
    if (ty < 0 || ty >= out.height()) continue;
    for (int dx = -radius; dx <= radius; ++dx) {
      const int tx = dst_x + dx;
      if (tx < 0 || tx >= out.width()) continue;
      const float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
      if (dist > r_f) continue;

      float weight = 1.0f;
      if (dist > inner_r) {
        const float t = (dist - inner_r) / std::max(1e-4f, r_f - inner_r);
        weight = 1.0f - t * t * (3.0f - 2.0f * t);
      }

      for (int c = 0; c < out.channels(); ++c) {
        const float patch_val =
            src.sample(src_x + dx, src_y + dy, c, BorderMode::ClampEdge);
        out.at(tx, ty, c) =
            (1.0f - weight) * out.at(tx, ty, c) + weight * patch_val;
      }
    }
  }
  return out;
}

Canvas median_filter(const Canvas& src, int radius) {
  const int r = std::max(1, radius);
  const int ksize = (2 * r + 1) * (2 * r + 1);
  Canvas out(src.width(), src.height(), src.channels());
  std::vector<float> window(static_cast<std::size_t>(ksize), 0.0f);

  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      for (int c = 0; c < src.channels(); ++c) {
        int idx = 0;
        for (int dy = -r; dy <= r; ++dy) {
          for (int dx = -r; dx <= r; ++dx) {
            window[static_cast<std::size_t>(idx++)] =
                src.sample(x + dx, y + dy, c, BorderMode::ClampEdge);
          }
        }
        std::nth_element(window.begin(), window.begin() + ksize / 2, window.end());
        out.at(x, y, c) = window[static_cast<std::size_t>(ksize / 2)];
      }
    }
  }
  return out;
}

} // namespace imglib
