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

static float mitchell_netravali_weight(float d, float b, float c) {
  const float ax = std::abs(d);
  const float ax2 = ax * ax;
  const float ax3 = ax2 * ax;
  if (ax < 1.0f) {
    return ((12.0f - 9.0f * b - 6.0f * c) * ax3 +
            (-18.0f + 12.0f * b + 6.0f * c) * ax2 + (6.0f - 2.0f * b)) /
           6.0f;
  }
  if (ax < 2.0f) {
    return ((-b - 6.0f * c) * ax3 + (6.0f * b + 30.0f * c) * ax2 +
            (-12.0f * b - 48.0f * c) * ax + (8.0f * b + 24.0f * c)) /
           6.0f;
  }
  return 0.0f;
}

Canvas resize_bicubic(const Canvas& src,
                      float scale_x,
                      float scale_y,
                      float b_param,
                      float c_param) {
  const float sy = (scale_y > 0.0f) ? scale_y : scale_x;
  const int out_w = std::max(1, static_cast<int>(std::floor(src.width() * scale_x)));
  const int out_h = std::max(1, static_cast<int>(std::floor(src.height() * sy)));
  Canvas out(out_w, out_h, src.channels());

  for (int y = 0; y < out_h; ++y) {
    const float src_y = static_cast<float>(y) / sy;
    const int iy0 = static_cast<int>(std::floor(src_y)) - 2;
    for (int x = 0; x < out_w; ++x) {
      const float src_x = static_cast<float>(x) / scale_x;
      const int ix0 = static_cast<int>(std::floor(src_x)) - 2;

      for (int ky = iy0; ky <= iy0 + 4; ++ky) {
        const float wy = mitchell_netravali_weight(src_y - ky, b_param, c_param);
        for (int kx = ix0; kx <= ix0 + 4; ++kx) {
          const float w = wy * mitchell_netravali_weight(src_x - kx, b_param, c_param);
          for (int ch = 0; ch < src.channels(); ++ch) {
            out.at(x, y, ch) += src.sample(kx, ky, ch, BorderMode::ClampEdge) * w;
          }
        }
      }
    }
  }
  return out;
}

static float lanczos_weight(float d, float a) {
  const float ax = std::abs(d);
  if (ax < 1e-6f) return 1.0f;
  if (ax >= a) return 0.0f;
  const float pix = kPi * d;
  return (a * std::sin(pix) * std::sin(pix / a)) / (pix * pix);
}

Canvas resize_lanczos(const Canvas& src, float scale_x, float scale_y, float lobes) {
  const float sy = (scale_y > 0.0f) ? scale_y : scale_x;
  const int out_w = std::max(1, static_cast<int>(std::floor(src.width() * scale_x)));
  const int out_h = std::max(1, static_cast<int>(std::floor(src.height() * sy)));
  Canvas out(out_w, out_h, src.channels());
  const int rad = static_cast<int>(std::ceil(lobes));

  for (int y = 0; y < out_h; ++y) {
    const float src_y = static_cast<float>(y) / sy;
    const int iy_center = static_cast<int>(std::floor(src_y));
    for (int x = 0; x < out_w; ++x) {
      const float src_x = static_cast<float>(x) / scale_x;
      const int ix_center = static_cast<int>(std::floor(src_x));

      for (int ky = iy_center - rad + 1; ky <= iy_center + rad; ++ky) {
        const float wy = lanczos_weight(src_y - ky, lobes);
        for (int kx = ix_center - rad + 1; kx <= ix_center + rad; ++kx) {
          const float w = wy * lanczos_weight(src_x - kx, lobes);
          for (int ch = 0; ch < src.channels(); ++ch) {
            out.at(x, y, ch) += src.sample(kx, ky, ch, BorderMode::ClampEdge) * w;
          }
        }
      }
    }
  }
  return out;
}

Canvas resize(const Canvas& src, float scale, FilterMode mode) {
  switch (mode) {
    case FilterMode::Nearest:
      return resize_nearest(src, scale);
    case FilterMode::Bicubic:
      return resize_bicubic(src, scale);
    case FilterMode::Lanczos3:
      return resize_lanczos(src, scale, -1.0f, 3.0f);
    case FilterMode::Bilinear:
    default:
      return resize_bilinear(src, scale);
  }
}

Canvas rotate_center(const Canvas& src, float angle_radians) {
  const float cx = 0.5f * static_cast<float>(src.width() - 1);
  const float cy = 0.5f * static_cast<float>(src.height() - 1);
  const float cos_t = std::cos(angle_radians);
  const float sin_t = std::sin(angle_radians);

  Canvas out(src.width(), src.height(), src.channels());
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float dx = static_cast<float>(x) - cx;
      const float dy = cy - static_cast<float>(y);
      const float sx = dx * cos_t + dy * sin_t + cx;
      const float sy = cy - (-dx * sin_t + dy * cos_t);
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = sample_bilinear(src, sx, sy, c, BorderMode::ZeroPad);
      }
    }
  }
  return out;
}

Canvas crop_region(const Canvas& src, int x0, int y0, int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("crop_region width and height must be positive");
  }
  Canvas out(width, height, src.channels());
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = src.sample(x0 + x, y0 + y, c, BorderMode::ClampEdge);
      }
    }
  }
  return out;
}

Canvas flip_horizontal(const Canvas& src) {
  Canvas out(src.width(), src.height(), src.channels());
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const int mx = src.width() - 1 - x;
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = src.at(mx, y, c);
      }
    }
  }
  return out;
}

Canvas flip_vertical(const Canvas& src) {
  Canvas out(src.width(), src.height(), src.channels());
  for (int y = 0; y < src.height(); ++y) {
    const int my = src.height() - 1 - y;
    for (int x = 0; x < src.width(); ++x) {
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = src.at(x, my, c);
      }
    }
  }
  return out;
}

Canvas apply_vignette(const Canvas& src, float strength, float power) {
  const float cx = 0.5f * static_cast<float>(src.width() - 1);
  const float cy = 0.5f * static_cast<float>(src.height() - 1);
  const float max_r = std::max(1.0f, std::sqrt(cx * cx + cy * cy));
  Canvas out(src.width(), src.height(), src.channels());

  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float dx = static_cast<float>(x) - cx;
      const float dy = static_cast<float>(y) - cy;
      const float r_norm = std::sqrt(dx * dx + dy * dy) / max_r;
      const float falloff =
          std::max(0.0f, 1.0f - strength * std::pow(r_norm, power));
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = src.at(x, y, c) * falloff;
      }
    }
  }
  return out;
}

} // namespace imglib
