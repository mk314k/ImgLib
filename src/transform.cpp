#include "imglib/transform.hpp"
#include <cmath>

namespace imglib {

static constexpr float kPi = 3.14159265358979323846f;

float sample_bilinear(const Canvas& src,
                      float x,
                      float y,
                      int c,
                      BorderMode border) {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float ax = x - static_cast<float>(x0);
  const float ay = y - static_cast<float>(y0);

  const float v00 = src.sample(x0, y0, c, border);
  const float v10 = src.sample(x1, y0, c, border);
  const float v01 = src.sample(x0, y1, c, border);
  const float v11 = src.sample(x1, y1, c, border);

  const float top = v00 * (1.0f - ax) + v10 * ax;
  const float bot = v01 * (1.0f - ax) + v11 * ax;
  return top * (1.0f - ay) + bot * ay;
}

Canvas resize_nearest(const Canvas& src, float scale_x, float scale_y) {
  const float sy = (scale_y > 0.0f) ? scale_y : scale_x;
  const int out_w = std::max(1, static_cast<int>(std::floor(src.width() * scale_x)));
  const int out_h = std::max(1, static_cast<int>(std::floor(src.height() * sy)));
  Canvas out(out_w, out_h, src.channels());

  for (int y = 0; y < out_h; ++y) {
    const int src_y = static_cast<int>(std::round(static_cast<float>(y) / sy));
    for (int x = 0; x < out_w; ++x) {
      const int src_x = static_cast<int>(std::round(static_cast<float>(x) / scale_x));
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = src.sample(src_x, src_y, c, BorderMode::ClampEdge);
      }
    }
  }
  return out;
}

Canvas resize_bilinear(const Canvas& src, float scale_x, float scale_y) {
  const float sy = (scale_y > 0.0f) ? scale_y : scale_x;
  const int out_w = std::max(1, static_cast<int>(std::floor(src.width() * scale_x)));
  const int out_h = std::max(1, static_cast<int>(std::floor(src.height() * sy)));
  Canvas out(out_w, out_h, src.channels());

  for (int y = 0; y < out_h; ++y) {
    const float src_y = static_cast<float>(y) / sy;
    for (int x = 0; x < out_w; ++x) {
      const float src_x = static_cast<float>(x) / scale_x;
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = sample_bilinear(src, src_x, src_y, c, BorderMode::ClampEdge);
      }
    }
  }
  return out;
}

} // namespace imglib
