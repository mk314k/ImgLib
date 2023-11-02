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

void Canvas::fill(float r, float g, float b) {
  const float col[4] = {r, g, b, 1.0f};
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      for (int c = 0; c < channels_; ++c) {
        buffer_[index(x, y, c)] = col[c];
      }
    }
  }
}

void Canvas::fill_rect(int x0, int y0, int x1, int y1, float r, float g, float b) {
  const int xs = std::max(0, std::min(x0, x1));
  const int xe = std::min(width_ - 1, std::max(x0, x1));
  const int ys = std::max(0, std::min(y0, y1));
  const int ye = std::min(height_ - 1, std::max(y0, y1));
  const float col[4] = {r, g, b, 1.0f};
  for (int y = ys; y <= ye; ++y) {
    for (int x = xs; x <= xe; ++x) {
      for (int c = 0; c < channels_; ++c) {
        buffer_[index(x, y, c)] = col[c];
      }
    }
  }
}

void Canvas::stroke_line(int x0, int y0, int x1, int y1, float r, float g, float b) {
  const float col[4] = {r, g, b, 1.0f};
  const int dx = x1 - x0;
  const int dy = y1 - y0;
  const int steps = std::max(std::abs(dx), std::abs(dy));
  if (steps == 0) {
    if (x0 >= 0 && x0 < width_ && y0 >= 0 && y0 < height_) {
      for (int c = 0; c < channels_; ++c) buffer_[index(x0, y0, c)] = col[c];
    }
    return;
  }
  for (int s = 0; s <= steps; ++s) {
    const float t = static_cast<float>(s) / static_cast<float>(steps);
    const int ix = static_cast<int>(std::round(x0 + t * dx));
    const int iy = static_cast<int>(std::round(y0 + t * dy));
    if (ix >= 0 && ix < width_ && iy >= 0 && iy < height_) {
      for (int c = 0; c < channels_; ++c) {
        buffer_[index(ix, iy, c)] = col[c];
      }
    }
  }
}

void ensure_same_shape(const Canvas& a, const Canvas& b) {
  if (a.width() != b.width() || a.height() != b.height() ||
      a.channels() != b.channels()) {
    throw std::invalid_argument("Canvas dimensions do not match");
  }
}

Canvas operator+(const Canvas& a, const Canvas& b) {
  ensure_same_shape(a, b);
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) + b.flat(i);
  return out;
}

Canvas operator-(const Canvas& a, const Canvas& b) {
  ensure_same_shape(a, b);
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) - b.flat(i);
  return out;
}

Canvas operator*(const Canvas& a, const Canvas& b) {
  ensure_same_shape(a, b);
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) * b.flat(i);
  return out;
}

Canvas operator/(const Canvas& a, const Canvas& b) {
  ensure_same_shape(a, b);
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::abs(b.flat(i)) < 1e-12f) {
      throw std::domain_error("Division by zero pixel in Canvas::operator/");
    }
    out.flat(i) = a.flat(i) / b.flat(i);
  }
  return out;
}

Canvas operator+(const Canvas& a, float scalar) {
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) + scalar;
  return out;
}

Canvas operator-(const Canvas& a, float scalar) {
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) - scalar;
  return out;
}

Canvas operator*(const Canvas& a, float scalar) {
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) * scalar;
  return out;
}

Canvas operator/(const Canvas& a, float scalar) {
  if (std::abs(scalar) < 1e-12f) {
    throw std::domain_error("Division by zero scalar in Canvas::operator/");
  }
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = a.flat(i) / scalar;
  return out;
}

Canvas operator+(float scalar, const Canvas& a) { return a + scalar; }
Canvas operator*(float scalar, const Canvas& a) { return a * scalar; }

Canvas operator-(float scalar, const Canvas& a) {
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) out.flat(i) = scalar - a.flat(i);
  return out;
}

Canvas operator/(float scalar, const Canvas& a) {
  Canvas out(a.width(), a.height(), a.channels());
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a.flat(i)) < 1e-12f) {
      throw std::domain_error("Division by zero pixel in scalar / Canvas");
    }
    out.flat(i) = scalar / a.flat(i);
  }
  return out;
}

} // namespace imglib
