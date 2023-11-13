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

Canvas canny_edges(const Canvas& src,
                   float sigma = 1.2f,
                   float low_thresh = 0.08f,
                   float high_thresh = 0.22f);

} // namespace imglib

#endif // IMGLIB_RETOUCH_HPP
