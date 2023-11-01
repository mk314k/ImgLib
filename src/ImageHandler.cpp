#include "../include/ImageHandler.h"

using namespace std;


float ImageHandler::uint8_to_float(const unsigned char &in) {
  return ((float)in) / (255.0f);
}

unsigned char ImageHandler::float_to_uint8(const float &in) {
  float out = in;
  if (out < 0)
    out = 0;
  if (out > 1)
    out = 1;
  return (unsigned char)(255.0f * out);
}

Image ImageHandler::imread(const std::string &filename) {
  std::vector<unsigned char> uint8_image;
  unsigned int h;
  unsigned int w;
  unsigned int c = 4;
  unsigned int oc = 3; // Throw away transparency

  // Decode PNG file In column major order with packed color values
  unsigned err = lodepng::decode(uint8_image, w, h, filename.c_str());
  if (err == 48) {
    throw FileNotFoundException();
  }

  Image img(h,w,oc);

  for (unsigned int x = 0; x < w; x++) {
    for (unsigned int y = 0; y < h; y++) {
      for (unsigned int z = 0; z < oc; z++) {
        img(x + y * w + z * w * h) =
            uint8_to_float(uint8_image[z + x * c + y * c * w]);
      }
    }
  }

}

void ImageHandler::imwrite(const Image &img, const std::string &filename) const {
  if (img.channels() != 1 && img.channels() != 3 && img.channels() != 4)
    throw ChannelException();
  int png_channels = 4;
  std::vector<unsigned char> uint8_image(img.height() * img.width() * png_channels,
                                         255);
  int z;
  for (int x = 0; x < img.width(); x++) {
    for (int y = 0; y < img.height(); y++) {
      for (z = 0; z < img.channels(); z++) {
        uint8_image[z + x * png_channels + y * png_channels * img.width()] =
            float_to_uint8(
                img(x + y * img.width() + z * img.width() * img.height()));
      }
      for (; z < 3; z++) { // Only executes when there is one channel
        uint8_image[z + x * png_channels + y * png_channels * img.width()] =
            float_to_uint8(
                img(x + y * img.width() + 0 * img.width() * img.height()));
      }
    }
  }
  lodepng::encode(filename.c_str(), uint8_image, img.width(), img.height());
}

void ImageHandler::debug_write(const Image &img) const {
  std::ostringstream ss;
  ss << "./Output/" << 1 << ".png";
  std::string filename = ss.str();
  imwrite(img, filename);
  //debugWriteNumber++;
}