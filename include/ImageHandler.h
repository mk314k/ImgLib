#ifndef __IMGHANDLER__H
#define __IMGHANDLER__H

#include <sstream>
#include "lodepng.h"
#include "ImageException.h"
#include "Image.h"


class ImageHandler{

public:
    Image imread(const std::string &filename);
    void imwrite(const Image &img, const std::string &filename) const;
    void debug_write(const Image &img) const;

    // Helper functions for reading and writing
  // Conversion Policy:
  //      uint8   float
  //      0       0.f
  //      255     1.f
    static float uint8_to_float(const unsigned char &in);
    static unsigned char float_to_uint8(const float &in);



};



#endif