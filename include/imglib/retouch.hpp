#ifndef IMGLIB_RETOUCH_HPP
#define IMGLIB_RETOUCH_HPP

#include "imglib/canvas.hpp"

namespace imglib {

Canvas clone_stamp(const Canvas& src,
                   int src_x,
                   int src_y,
                   int dst_x,
                   int dst_y,
                   int radius,
                   float hardness = 0.6f);

Canvas median_filter(const Canvas& src, int radius = 1);

} // namespace imglib

#endif // IMGLIB_RETOUCH_HPP
