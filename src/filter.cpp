#include "imglib/filter.hpp"
#include "imglib/color.hpp"
#include <cmath>

namespace imglib {

Kernel2D::Kernel2D(int width, int height, float fill_val)
    : width_(width), height_(height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Kernel2D dimensions must be positive");
  }
  weights_.assign(static_cast<std::size_t>(width) * height, fill_val);
}

Kernel2D::Kernel2D(int width, int height, const std::vector<float>& weights)
    : width_(width), height_(height), weights_(weights) {
  if (width <= 0 || height <= 0 ||
      weights.size() != static_cast<std::size_t>(width) * height) {
    throw std::invalid_argument("Kernel2D weights size mismatch");
  }
}

float& Kernel2D::operator()(int x, int y) {
  if (x < 0 || x >= width_ || y < 0 || y >= height_) {
    throw std::out_of_range("Kernel2D coordinate out of bounds");
  }
  return weights_[static_cast<std::size_t>(y) * width_ + x];
}

const float& Kernel2D::operator()(int x, int y) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_) {
    throw std::out_of_range("Kernel2D coordinate out of bounds");
  }
  return weights_[static_cast<std::size_t>(y) * width_ + x];
}

Kernel2D Kernel2D::transposed() const {
  Kernel2D out(height_, width_);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      out(y, x) = (*this)(x, y);
    }
  }
  return out;
}

Canvas Kernel2D::convolve(const Canvas& src, BorderMode border) const {
  Canvas out(src.width(), src.height(), src.channels());
  const int rx = width_ / 2;
  const int ry = height_ / 2;

  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      for (int c = 0; c < src.channels(); ++c) {
        float acc = 0.0f;
        for (int ky = 0; ky < height_; ++ky) {
          for (int kx = 0; kx < width_; ++kx) {
            acc += (*this)(kx, ky) *
                   src.sample(x - kx + rx, y - ky + ry, c, border);
          }
        }
        out.at(x, y, c) = acc;
      }
    }
  }
  return out;
}

std::vector<float> gaussian_kernel_1d(float sigma, float truncate) {
  const int radius = std::max(1, static_cast<int>(std::ceil(sigma * truncate)));
  const int ksize = 2 * radius + 1;
  std::vector<float> k(static_cast<std::size_t>(ksize), 0.0f);
  float sum = 0.0f;
  for (int i = 0; i < ksize; ++i) {
    const float d = static_cast<float>(i - radius);
    k[static_cast<std::size_t>(i)] = std::exp(-(d * d) / (2.0f * sigma * sigma));
    sum += k[static_cast<std::size_t>(i)];
  }
  for (float& w : k) w /= sum;
  return k;
}

std::vector<float> gaussian_kernel_2d(float sigma, float truncate) {
  const int radius = std::max(1, static_cast<int>(std::ceil(sigma * truncate)));
  const int ksize = 2 * radius + 1;
  std::vector<float> k(static_cast<std::size_t>(ksize) * ksize, 0.0f);
  float sum = 0.0f;
  for (int y = 0; y < ksize; ++y) {
    for (int x = 0; x < ksize; ++x) {
      const float dx = static_cast<float>(x - radius);
      const float dy = static_cast<float>(y - radius);
      const float w = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
      k[static_cast<std::size_t>(y) * ksize + x] = w;
      sum += w;
    }
  }
  for (float& w : k) w /= sum;
  return k;
}

Canvas box_blur(const Canvas& src, int radius, BorderMode border) {
  const int ksize = 2 * std::max(0, radius) + 1;
  const float weight = 1.0f / static_cast<float>(ksize * ksize);
  Kernel2D k(ksize, ksize, weight);
  return k.convolve(src, border);
}

Canvas gaussian_blur_separable(const Canvas& src,
                               float sigma,
                               float truncate,
                               BorderMode border) {
  const std::vector<float> w1d = gaussian_kernel_1d(sigma, truncate);
  const int ksize = static_cast<int>(w1d.size());
  Kernel2D kx(ksize, 1, w1d);
  Kernel2D ky(1, ksize, w1d);
  return ky.convolve(kx.convolve(src, border), border);
}

Canvas gaussian_blur_2d(const Canvas& src,
                        float sigma,
                        float truncate,
                        BorderMode border) {
  const std::vector<float> w2d = gaussian_kernel_2d(sigma, truncate);
  const int ksize = static_cast<int>(std::round(std::sqrt(w2d.size())));
  Kernel2D k(ksize, ksize, w2d);
  return k.convolve(src, border);
}

Canvas unsharp_mask(const Canvas& src,
                    float sigma,
                    float strength,
                    float truncate) {
  const Canvas blurred = gaussian_blur_separable(src, sigma, truncate);
  const Canvas detail = src - blurred;
  return src + detail * strength;
}

Canvas sobel_x(const Canvas& src, BorderMode border) {
  const Kernel2D kx(3, 3, {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f});
  return kx.convolve(src, border);
}

Canvas sobel_y(const Canvas& src, BorderMode border) {
  const Kernel2D ky(3, 3, {-1.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f});
  return ky.convolve(src, border);
}

Canvas sobel_magnitude(const Canvas& src, BorderMode border) {
  const Canvas gx = sobel_x(src, border);
  const Canvas gy = sobel_y(src, border);
  Canvas mag = gx * gx + gy * gy;
  for (std::size_t i = 0; i < mag.size(); ++i) {
    mag.flat(i) = std::sqrt(mag.flat(i));
  }
  return mag;
}

Canvas bilateral_filter(const Canvas& src,
                        float sigma_range,
                        float sigma_spatial,
                        float truncate,
                        BorderMode border) {
  const int radius = std::max(1, static_cast<int>(std::ceil(sigma_spatial * truncate)));
  Canvas out(src.width(), src.height(), src.channels());

  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      std::vector<float> accum(static_cast<std::size_t>(src.channels()), 0.0f);
      float norm = 0.0f;

      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
          float range_d2 = 0.0f;
          for (int c = 0; c < src.channels(); ++c) {
            const float diff = src.at(x, y, c) - src.sample(x + dx, y + dy, c, border);
            range_d2 += diff * diff;
          }
          const float w_spatial = std::exp(
              -static_cast<float>(dx * dx + dy * dy) / (2.0f * sigma_spatial * sigma_spatial));
          const float w_range = std::exp(-range_d2 / (2.0f * sigma_range * sigma_range));
          const float w = w_spatial * w_range;
          norm += w;
          for (int c = 0; c < src.channels(); ++c) {
            accum[static_cast<std::size_t>(c)] +=
                w * src.sample(x + dx, y + dy, c, border);
          }
        }
      }
      for (int c = 0; c < src.channels(); ++c) {
        out.at(x, y, c) = accum[static_cast<std::size_t>(c)] / std::max(1e-8f, norm);
      }
    }
  }
  return out;
}

Canvas bilateral_yuv_filter(const Canvas& src,
                            float sigma_range,
                            float sigma_y,
                            float sigma_uv,
                            float truncate) {
  Canvas yuv = rgb_to_yuv(src);
  const Canvas filt_y = bilateral_filter(yuv, sigma_range, sigma_y, truncate);
  const Canvas filt_uv = bilateral_filter(yuv, sigma_range, sigma_uv, truncate);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      yuv.at(x, y, 0) = filt_y.at(x, y, 0);
      yuv.at(x, y, 1) = filt_uv.at(x, y, 1);
      yuv.at(x, y, 2) = filt_uv.at(x, y, 2);
    }
  }
  return yuv_to_rgb(yuv);
}

Canvas max_pool_filter(const Canvas& src, int window_diameter) {
  const int lo = window_diameter / 2;
  const int hi = window_diameter - lo - 1;
  Canvas out(src.width(), src.height(), src.channels());

  for (int y = lo; y < src.height() - hi; ++y) {
    for (int x = lo; x < src.width() - hi; ++x) {
      for (int c = 0; c < src.channels(); ++c) {
        float best = src.at(x, y, c);
        for (int dy = -lo; dy <= hi; ++dy) {
          for (int dx = -lo; dx <= hi; ++dx) {
            best = std::max(best, src.at(x + dx, y + dy, c));
          }
        }
        out.at(x, y, c) = best;
      }
    }
  }
  return out;
}

Canvas unit_impulse(int size) {
  Canvas out(size, size, 1, 0.0f);
  out.at(size / 2, size / 2, 0) = 1.0f;
  return out;
}

} // namespace imglib
