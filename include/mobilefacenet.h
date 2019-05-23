#ifndef __MOBILEFACENET_H
#define __MOBILEFACENET_H

#include <opencv/cv.h>
#include "darknet.h"
#include "utiles.h"

#define H 112
#define W 96
#define N 128

network* load_mobilefacenet();
image convert_mobilefacenet_image(image im);
int verify(network* net, image im1, image im2, float thresh);

#endif