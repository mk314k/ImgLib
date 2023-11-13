#include "imglib/retouch.hpp"
#include "imglib/color.hpp"
#include "imglib/filter.hpp"
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

Canvas canny_edges(const Canvas& src,
                   float sigma,
                   float low_thresh,
                   float high_thresh) {
  const Canvas smooth = gaussian_blur_separable(to_grayscale(src), sigma);
  const Canvas gx = sobel_x(smooth);
  const Canvas gy = sobel_y(smooth);

  const int w = smooth.width();
  const int h = smooth.height();
  Canvas mag(w, h, 1, 0.0f);
  std::vector<int> dir_sector(static_cast<std::size_t>(w) * h, 0);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float ix = gx.at(x, y, 0);
      const float iy = gy.at(x, y, 0);
      mag.at(x, y, 0) = std::sqrt(ix * ix + iy * iy);

      float deg = std::atan2(iy, ix) * (180.0f / 3.14159265358979323846f);
      if (deg < 0.0f) deg += 180.0f;
      int sector = 0;
      if (deg >= 22.5f && deg < 67.5f) sector = 1;
      else if (deg >= 67.5f && deg < 112.5f) sector = 2;
      else if (deg >= 112.5f && deg < 157.5f) sector = 3;
      dir_sector[static_cast<std::size_t>(y) * w + x] = sector;
    }
  }

  // Non-maximum suppression
  Canvas nms(w, h, 1, 0.0f);
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      const float m = mag.at(x, y, 0);
      const int sec = dir_sector[static_cast<std::size_t>(y) * w + x];
      float n1 = 0.0f, n2 = 0.0f;
      if (sec == 0) {
        n1 = mag.at(x - 1, y, 0);
        n2 = mag.at(x + 1, y, 0);
      } else if (sec == 1) {
        n1 = mag.at(x - 1, y - 1, 0);
        n2 = mag.at(x + 1, y + 1, 0);
      } else if (sec == 2) {
        n1 = mag.at(x, y - 1, 0);
        n2 = mag.at(x, y + 1, 0);
      } else {
        n1 = mag.at(x - 1, y + 1, 0);
        n2 = mag.at(x + 1, y - 1, 0);
      }
      if (m >= n1 && m >= n2) {
        nms.at(x, y, 0) = m;
      }
    }
  }

  // Double threshold + 8-connected hysteresis
  Canvas edges(w, h, 1, 0.0f);
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      const float v = nms.at(x, y, 0);
      if (v >= high_thresh) {
        edges.at(x, y, 0) = 1.0f;
      } else if (v >= low_thresh) {
        bool connected = false;
        for (int dy = -1; dy <= 1 && !connected; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (nms.at(x + dx, y + dy, 0) >= high_thresh) {
              connected = true;
              break;
            }
          }
        }
        edges.at(x, y, 0) = connected ? 1.0f : 0.0f;
      }
    }
  }
  return edges;
}

} // namespace imglib
