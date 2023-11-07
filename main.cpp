
#include "include/Image.h"
#include "include/ImageManipulation.h"
#include "include/filtering.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "include/ImageHandler.h"

using namespace std;

Image maskThreshold(const Image &mask){
  Image res = Image(mask.width(),mask.height(),3);
  float val =0.0;
  for (int x=0;x<mask.width();x++){
      for (int y=0; y<mask.height();y++){
        if (mask(x,y)>=0.5){
          val=1.0;
        }else{
          val = 0.0;
        }
        for (int z=0;z<3;z++){
            res(x,y,z) = val;
        }
      }
  }
  return res;
}
void imageCopy(const Image &s, Image &d, int y1, int x1, int y2=0, int x2=0){
  int w = min(d.width(),s.width());
  int h = min(d.height(),s.height());
  for (int x=0;x<w;x++){
    for (int y=0; y<h;y++){
      for (int z=0;z<d.channels();z++){
        d(x+x2,y+y2,z) = s(x+x1,y+y1,z);
      }
    }
  }
}

int main() {

  return 0;
}
