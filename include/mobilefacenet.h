#ifndef __MOBILEFACENET_H
#define __MOBILEFACENET_H

#include <opencv/cv.h>
#include "darknet.h"
#include "mtcnn.h"
#include "util.h"

#define H 112
#define W 96
#define N 128

/*
 * * coefficients for Logistic Regression
 *      y = sigmoid(WEIGHT * cosion + BIAS)
 */
#define WEIGHT 11.7241106
#define BIAS -3.68407159

landmark initAligned();
network* load_mobilefacenet();
image convert_mobilefacenet_image(image im);
int verify(network* net, image im1, image im2, float* cosine);

int verify_input_images(int argc, char** argv);
int verify_lfw_images(int argc, char** argv);

#endif