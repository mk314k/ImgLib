#include "imglib/color.hpp"
#include <cmath>

namespace imglib {

Canvas adjust_brightness(const Canvas& src, float gain) {
  return src * gain;
}

Canvas adjust_contrast(const Canvas& src, float factor, float pivot) {
  return (src - pivot) * factor + pivot;
}

Canvas to_grayscale(const Canvas& src, float wr, float wg, float wb) {
  if (src.channels() == 1) return src;
  const float norm = (wr + wg + wb != 0.0f) ? (wr + wg + wb) : 1.0f;
  Canvas gray(src.width(), src.height(), 1);
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float val =
          (src.at(x, y, 0) * wr + src.at(x, y, 1) * wg + src.at(x, y, 2) * wb) /
          norm;
      gray.at(x, y, 0) = val;
    }
  }
  return gray;
}

LumaChromaPair split_luma_chroma(const Canvas& src) {
  Canvas luma = to_grayscale(src);
  Canvas chroma(src.width(), src.height(), src.channels());
  for (int y = 0; y < src.height(); ++y) {
    for (int x = 0; x < src.width(); ++x) {
      const float l_val = luma.at(x, y, 0);
      const float safe_l = (l_val > 1e-6f) ? l_val : 1e-6f;
      for (int c = 0; c < src.channels(); ++c) {
        chroma.at(x, y, c) = (l_val > 1e-6f) ? (src.at(x, y, c) / safe_l) : 0.0f;
      }
    }
  }
  return {luma, chroma};
}

Canvas merge_luma_chroma(const LumaChromaPair& pair) {
  Canvas out(pair.chroma.width(), pair.chroma.height(), pair.chroma.channels());
  for (int y = 0; y < out.height(); ++y) {
    for (int x = 0; x < out.width(); ++x) {
      const float l_val = pair.luma.at(x, y, 0);
      for (int c = 0; c < out.channels(); ++c) {
        out.at(x, y, c) = pair.chroma.at(x, y, c) * l_val;
      }
    }
  }
  return out;
}

Canvas adjust_luma_tone(const Canvas& src,
                        float brightness_gain,
                        float contrast_gain,
                        float pivot) {
  LumaChromaPair lc = split_luma_chroma(src);
  lc.luma = adjust_brightness(lc.luma, brightness_gain);
  lc.luma = adjust_contrast(lc.luma, contrast_gain, pivot);
  return merge_luma_chroma(lc);
}

} // namespace imglib
