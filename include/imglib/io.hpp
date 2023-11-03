#ifndef IMGLIB_IO_HPP
#define IMGLIB_IO_HPP

#include "imglib/canvas.hpp"
#include <string>

namespace imglib {

Canvas load_png(const std::string& filepath);
void save_png(const Canvas& canvas, const std::string& filepath);

Canvas load_ppm(const std::string& filepath);
void save_ppm(const Canvas& canvas, const std::string& filepath);

unsigned char unit_to_byte(float val);
float byte_to_unit(unsigned char b);

} // namespace imglib

#endif // IMGLIB_IO_HPP
