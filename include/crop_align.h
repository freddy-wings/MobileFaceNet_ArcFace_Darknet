/*
 * @Author: louishsu
 * @Date: 2019-05-31 11:30:23 
 * @Last Modified by:   louishsu 
 * @Last Modified time: 2019-05-31 11:30:23 
 */
#ifndef __CROP_ALIGN_H
#define __CROP_ALIGN_H

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "darknet.h"
#include "mtcnn.h"

landmark initAligned();
image crop_image_by_box(image im, bbox a, int h, int w);
landmark substract_bias(landmark mark, float x, float y);
image align_image_with_landmark(image im, landmark src, landmark dst);

#endif