#include "imglib/io.hpp"
#include "lodepng.h"
#include <fstream>
#include <sstream>

namespace imglib {

unsigned char unit_to_byte(float val) {
  const float clamped = std::max(0.0f, std::min(1.0f, val));
  return static_cast<unsigned char>(std::round(clamped * 255.0f));
}

float byte_to_unit(unsigned char b) {
  return static_cast<float>(b) / 255.0f;
}

Canvas load_png(const std::string& filepath) {
  std::vector<unsigned char> rgba;
  unsigned int w = 0;
  unsigned int h = 0;
  const unsigned int err = lodepng::decode(rgba, w, h, filepath.c_str());
  if (err != 0) {
    throw std::runtime_error(std::string("PNG decode error (") + filepath +
                             "): " + lodepng_error_text(err));
  }
  Canvas out(static_cast<int>(w), static_cast<int>(h), 3);
  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      const std::size_t base = (static_cast<std::size_t>(y) * w + x) * 4u;
      for (int c = 0; c < 3; ++c) {
        out.at(static_cast<int>(x), static_cast<int>(y), c) =
            byte_to_unit(rgba[base + static_cast<std::size_t>(c)]);
      }
    }
  }
  return out;
}

void save_png(const Canvas& canvas, const std::string& filepath) {
  if (canvas.empty()) {
    throw std::invalid_argument("Cannot save empty Canvas to PNG");
  }
  const int w = canvas.width();
  const int h = canvas.height();
  const int ch = canvas.channels();
  std::vector<unsigned char> rgba(static_cast<std::size_t>(w) * h * 4u, 255u);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const std::size_t base = (static_cast<std::size_t>(y) * w + x) * 4u;
      if (ch == 1) {
        const unsigned char gray = unit_to_byte(canvas.at(x, y, 0));
        rgba[base + 0] = gray;
        rgba[base + 1] = gray;
        rgba[base + 2] = gray;
      } else {
        for (int c = 0; c < 3; ++c) {
          rgba[base + static_cast<std::size_t>(c)] =
              unit_to_byte(canvas.at(x, y, c));
        }
        if (ch == 4) {
          rgba[base + 3] = unit_to_byte(canvas.at(x, y, 3));
        }
      }
    }
  }

  const unsigned int err = lodepng::encode(
      filepath.c_str(), rgba, static_cast<unsigned int>(w),
      static_cast<unsigned int>(h));
  if (err != 0) {
    throw std::runtime_error(std::string("PNG encode error (") + filepath +
                             "): " + lodepng_error_text(err));
  }
}

void save_ppm(const Canvas& canvas, const std::string& filepath) {
  std::ofstream out(filepath, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Unable to open PPM output file: " + filepath);
  }
  out << "P6\n" << canvas.width() << " " << canvas.height() << "\n255\n";
  for (int y = 0; y < canvas.height(); ++y) {
    for (int x = 0; x < canvas.width(); ++x) {
      unsigned char rgb[3];
      if (canvas.channels() == 1) {
        rgb[0] = rgb[1] = rgb[2] = unit_to_byte(canvas.at(x, y, 0));
      } else {
        rgb[0] = unit_to_byte(canvas.at(x, y, 0));
        rgb[1] = unit_to_byte(canvas.at(x, y, 1));
        rgb[2] = unit_to_byte(canvas.at(x, y, 2));
      }
      out.write(reinterpret_cast<const char*>(rgb), 3);
    }
  }
}

Canvas load_ppm(const std::string& filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to open PPM input file: " + filepath);
  }
  std::string magic;
  int w = 0, h = 0, maxv = 0;
  in >> magic >> w >> h >> maxv;
  if (magic != "P6" || w <= 0 || h <= 0 || maxv <= 0) {
    throw std::runtime_error("Invalid binary P6 PPM header: " + filepath);
  }
  in.get(); // consume single whitespace byte after maxv
  Canvas out(w, h, 3);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      unsigned char rgb[3] = {0, 0, 0};
      in.read(reinterpret_cast<char*>(rgb), 3);
      out.at(x, y, 0) = byte_to_unit(rgb[0]);
      out.at(x, y, 1) = byte_to_unit(rgb[1]);
      out.at(x, y, 2) = byte_to_unit(rgb[2]);
    }
  }
  return out;
}

} // namespace imglib
