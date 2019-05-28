#include "mobilefacenet.h"
#include "mtcnn.h"
#include "parser.h"
#include "activations.h"

network* load_mobilefacenet()
{
    network* net = load_network("cfg/mobilefacenet.cfg", 
                    "weights/mobilefacenet.weights", 0);
    return net;
}

landmark initAligned()
{
    landmark aligned = {0};

    aligned.x1 = 30.2946;
    aligned.y1 = 51.6963;
    aligned.x2 = 65.5318;
    aligned.y2 = 51.5014;
    aligned.x3 = 48.0252;
    aligned.y3 = 71.7366;
    aligned.x4 = 33.5493;
    aligned.y4 = 92.3655;
    aligned.x5 = 62.7299;
    aligned.y5 = 92.2041;

    return aligned;
}

/*
 * Args:
 *      im: {image} RGB image, range[0, 1]
 * Returns:
 *      cvt:{image} BGR image, range[-1, 1]
 * */
image convert_mobilefacenet_image(image im)
{
    int size = im.h*im.w*im.c;

    image cvt = copy_image(im);         // RGB, 0~1
    for (int i = 0; i < size; i++ ){
        float val = im.data[i]*255.;
        val = (val - 127.5) / 128.;
        cvt.data[i] = val; 
    }

    rgbgr_image(cvt);                   // BGR, -1~1
    return cvt;
}

/*
 * Args:
 *      net:    {network*}  MobileFaceNet
 *      im1/2:  {image}     image of size `3 x H x W`
 *      cosine: {float*}    threshold of verification, will be replaced with cosion distance.
 * Returns:
 *      isOne:  {int}       if the same, return 1; else 0.
 * */
int verify(network* net, image im1, image im2, float* cosine)
{
    assert(im1.w == W && im1.h == H);
    assert(im2.w == W && im2.h == H);

    float* X;

    float* feat1 = calloc(N*2, sizeof(float));
    image cvt1 = convert_mobilefacenet_image(im1);
    X = network_predict(net, cvt1.data);
    memcpy(feat1, X, N*sizeof(float));
    flip_image(cvt1);
    X = network_predict(net, cvt1.data);
    memcpy(feat1 + N, X, N*sizeof(float));
    
    float* feat2 = calloc(N*2, sizeof(float));
    image cvt2 = convert_mobilefacenet_image(im2);
    X = network_predict(net, cvt2.data);
    memcpy(feat2, X, N*sizeof(float));
    flip_image(cvt2);
    X = network_predict(net, cvt2.data);
    memcpy(feat2 + N, X, N*sizeof(float));

    float dist = distCosine(feat1, feat2, N*2);
    int is_one = -1;  

    // if (dist < *cosine){
    //     is_one = 0;
    // } else {
    //     is_one = 1;
    // }

    is_one = logistic_activate(WEIGHT*dist + BIAS) > 0.5? 1: 0;
    // printf("%.6f, %.6f\n", coef_*dist, logistic_activate(coef_*dist));
    
    *cosine = dist;

    free(feat1); free(feat2);
    free_image(cvt1); free_image(cvt2);

    return is_one;
}

