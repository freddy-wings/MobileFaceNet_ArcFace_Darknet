#include "mobilefacenet.h"
#include "mtcnn.h"
#include "parser.h"

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
}

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

int verify(network* net, image im1, image im2, float thresh)
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
    printf("cosine: %f\n", dist);
    
    if (dist < thresh){
        return 0;
    } else {
        return 1;
    }

    free(feat1); free(feat2);
    free_image(cvt1); free_image(cvt2);
}

