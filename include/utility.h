#ifndef __UTILITY__H
#define __UTILITY__H

#include <sstream>
#include "lodepng.h"
#include "ImageException.h"
#include "Image.h"


class utils{

public:
    Image imread(const std::string &filename);
    


};



#endif