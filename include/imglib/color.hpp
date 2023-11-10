#ifndef IMGLIB_COLOR_HPP
#define IMGLIB_COLOR_HPP

#include "imglib/canvas.hpp"
#include <utility>

namespace imglib {

struct LumaChromaPair {
  Canvas luma;
  Canvas chroma;
};

Canvas adjust_brightness(const Canvas& src, float gain);
Canvas adjust_contrast(const Canvas& src, float factor, float pivot = 0.5f);

Canvas to_grayscale(const Canvas& src,
                    float wr = 0.299f,
                    float wg = 0.587f,
                    float wb = 0.114f);

LumaChromaPair split_luma_chroma(const Canvas& src);
Canvas merge_luma_chroma(const LumaChromaPair& pair);
Canvas adjust_luma_tone(const Canvas& src,
                        float brightness_gain,
                        float contrast_gain,
                        float pivot = 0.3f);

Canvas rgb_to_yuv(const Canvas& src);
Canvas yuv_to_rgb(const Canvas& src);
Canvas adjust_saturation(const Canvas& src, float scale);

Canvas apply_gamma(const Canvas& src, float gamma);
Canvas quantize_bits(const Canvas& src, int bits_per_channel);

Canvas gray_world_balance(const Canvas& src);
std::pair<Canvas, Canvas> opponent_afterimage_pair(const Canvas& src);

Canvas rgb_to_hsl(const Canvas& src);
Canvas hsl_to_rgb(const Canvas& src);
Canvas shift_hsl(const Canvas& src,
                 float delta_hue_deg,
                 float sat_scale = 1.0f,
                 float light_offset = 0.0f);

Canvas adjust_levels(const Canvas& src,
                     float in_black = 0.0f,
                     float in_white = 1.0f,
                     float gamma = 1.0f,
                     float out_black = 0.0f,
                     float out_white = 1.0f);

Canvas equalize_histogram(const Canvas& src);

} // namespace imglib

#endif // IMGLIB_COLOR_HPP
