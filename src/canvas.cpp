#include "imglib/canvas.hpp"
#include <cfloat>

namespace imglib {

Canvas::Canvas(int width, int height, int channels, float init_val)
    : width_(width), height_(height), channels_(channels) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Canvas width and height must be positive");
  }
  if (channels != 1 && channels != 3 && channels != 4) {
    throw std::invalid_argument("Canvas channels must be 1, 3, or 4");
  }
  const std::size_t total =
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) *
      static_cast<std::size_t>(channels_);
  buffer_.assign(total, init_val);
}

float& Canvas::at(int x, int y, int c) {
  if (x < 0 || x >= width_ || y < 0 || y >= height_ || c < 0 || c >= channels_) {
    throw std::out_of_range("Canvas coordinate out of bounds");
  }
  return buffer_[index(x, y, c)];
}

const float& Canvas::at(int x, int y, int c) const {
  if (x < 0 || x >= width_ || y < 0 || y >= height_ || c < 0 || c >= channels_) {
    throw std::out_of_range("Canvas coordinate out of bounds");
  }
  return buffer_[index(x, y, c)];
}

float& Canvas::flat(std::size_t idx) {
  if (idx >= buffer_.size()) {
    throw std::out_of_range("Canvas flat index out of bounds");
  }
  return buffer_[idx];
}

const float& Canvas::flat(std::size_t idx) const {
  if (idx >= buffer_.size()) {
    throw std::out_of_range("Canvas flat index out of bounds");
  }
  return buffer_[idx];
}

static int reflect_coord(int v, int limit) {
  if (limit <= 1) return 0;
  while (v < 0 || v >= limit) {
    if (v < 0) v = -v;
    if (v >= limit) v = 2 * limit - 2 - v;
  }
  return v;
}

float Canvas::sample(int x, int y, int c, BorderMode mode, float pad_val) const {
  if (c < 0 || c >= channels_) {
    throw std::out_of_range("Canvas channel out of bounds");
  }
  if (x >= 0 && x < width_ && y >= 0 && y < height_) {
    return buffer_[index(x, y, c)];
  }
  if (mode == BorderMode::ZeroPad) {
    return pad_val;
  }
  if (mode == BorderMode::Reflect) {
    const int rx = reflect_coord(x, width_);
    const int ry = reflect_coord(y, height_);
    return buffer_[index(rx, ry, c)];
  }
  const int cx = std::max(0, std::min(width_ - 1, x));
  const int cy = std::max(0, std::min(height_ - 1, y));
  return buffer_[index(cx, cy, c)];
}

float Canvas::min_value() const {
  if (buffer_.empty()) return 0.0f;
  return *std::min_element(buffer_.begin(), buffer_.end());
}

float Canvas::max_value() const {
  if (buffer_.empty()) return 0.0f;
  return *std::max_element(buffer_.begin(), buffer_.end());
}

Canvas Canvas::clamped(float lo, float hi) const {
  Canvas out(width_, height_, channels_);
  for (std::size_t i = 0; i < buffer_.size(); ++i) {
    out.buffer_[i] = std::max(lo, std::min(hi, buffer_[i]));
  }
  return out;
}

} // namespace imglib
