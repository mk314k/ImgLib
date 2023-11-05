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

} // namespace imglib
