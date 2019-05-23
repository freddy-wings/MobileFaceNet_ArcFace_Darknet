#include "mobilefacenet.h"

network* load_mobilefacenet()
{
    network* net = load_network("cfg/mobilefacenet.cfg", 
                    "weights/mobilefacenet.weights", 0);
    return net;
}

image convert_mobilefacenet_image(image im)
{
    image cvt = copy_image(im);         // RGB, 0~1
    int size = im.h*im.w*im.c;
    for (int i = 0; i < size; i++ ){
        float val = im.data[i]*255;
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

    layer l = {0};

    image cvt1 = convert_mobilefacenet_image(im1);
    network_predict(net, cvt1.data);
    float* feat1 = calloc(N, sizeof(float));
    l = net->layers[net->n-1];
    memcpy(feat1, l.output, N);
    free_image(cvt1);
    
    image cvt2 = convert_mobilefacenet_image(im2);
    network_predict(net, cvt2.data);
    float* feat2 = calloc(N, sizeof(float));
    l = net->layers[net->n-1];
    memcpy(feat2, l.output, N);
    free_image(cvt2);

    float dist = distCosine(feat1, feat2, N);
    free(feat1); free(feat2);

    if (dist < thresh){
        return 0;
    } else {
        return 1;
    }
}

