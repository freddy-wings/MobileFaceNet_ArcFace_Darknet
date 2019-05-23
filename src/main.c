#include "mtcnn.h"
#include "mobilefacenet.h"

image crop_image_by_box(image im, bbox a, int h, int w)
{
    float cx = (a.x2 + a.x1) / 2;   // centroid
    float cy = (a.y2 + a.y1) / 2;

    // padding
    float w_ = a.x2 - a.x1 + 1;
    float h_ = a.y2 - a.y1 + 1;
    float ratio_src = h_ / w_;
    float ratio_dst = (float)h / (float)w;
    int ww_ = 0, hh_ = 0;
    if (ratio_src < ratio_dst){
        ww_ = (int)w_; hh_ = (int)(ww_ * ratio_dst); // 在行方向上进行填充,列数(w_)保持不变
    } else {
        hh_ = (int)h_; ww_ = (int)(hh_ / ratio_dst); // 在列方向上进行填充,行数(h_)保持不变
    }

    int x1 = (int)a.x1;
    int x2 = (int)a.x2;
    int y1 = (int)a.y1; 
    int y2 = (int)a.y2;
    int xx1 = 0, yy1 = 0;
    int xx2 = ww_ - 1;
    int yy2 = hh_ - 1;
    if (x1 < 0){xx1 = - x1; x1 = 0;}
    if (y1 < 0){yy1 = - y1; y1 = 0;}
    if (x2 > im.w - 1){xx2 = (x2-x1+1) + im.w - x2 - 2; x2 = im.w - 1;}
    if (y2 > im.h - 1){yy2 = (y2-y1+1) + im.h - y2 - 2; y2 = im.h - 1;}
    
    // crop
    image crop = make_image(ww_, hh_, im.c);
    for (int k = 0; k < im.c; k++ ){
        for (int j = yy1; j < yy2 + 1; j++ ){
            for (int i = xx1; i < xx2 + 1; i++ ){
                int x = x1 + i; int y = y1 + j;
                float val = im.data[x + y*im.w + k*im.w*im.h];
                crop.data[i + j*ww_ + k*ww_*hh_] = val;
            }
        }
    }

    image resized = resize_image(crop, w, h);
    free_image(crop);
    return resized;
}


#if 0
int main(int argc, char** argv)
{
    // ======================= LOAD IMAGES ======================== //
    char* imgpath1 = find_char_arg(argc, argv, "--image1", "images/Aaron_Eckhart_0001.jpg");
    // char* imgpath1 = find_char_arg(argc, argv, "--image1", "images/Aaron_Peirsol_0001.jpg");
    char* imgpath2 = find_char_arg(argc, argv, "--image2", "images/Aaron_Peirsol_0002.jpg");
    image im1 = load_image_color(imgpath1, 0, 0); 
    image im2 = load_image_color(imgpath2, 0, 0); 
    show_image(im1, "im1", 500); show_image(im2, "im2", 500);

    // ======================== INITIALIZE ======================== //
    params p = initParams(argc, argv);
    network* pnet = load_mtcnn_net("PNet");
    network* rnet = load_mtcnn_net("RNet");
    network* onet = load_mtcnn_net("ONet");
    network* mobilefacenet = load_mobilefacenet();
    
    // ======================== MTCNN STEP ======================== //
    // detect image1
    detect* dets = calloc(0, sizeof(dets)); int n = 0;
    detect_image(pnet, rnet, onet, im1, &n, &dets, p);
    show_detect(im1, dets, n, "detect1", 500, 1, 1, 1);
    if (n == 0){printf("image 1 not detected!\n"); return -1;}
    int idx1 = keep_one(dets, n, im1);
    image crop1 = crop_image_by_box(im1, dets[idx1].bx, H, W);

    // detect image2
    dets = realloc(dets, 0); n = 0;
    detect_image(pnet, rnet, onet, im2, &n, &dets, p);
    show_detect(im2, dets, n, "detect2", 500, 1, 1, 1);
    if (n == 0){printf("image 2 not detected!\n"); return -1;}
    int idx2 = keep_one(dets, n, im2);
    image crop2 = crop_image_by_box(im2, dets[idx2].bx, H, W);

    // ==================== MOBILEFACENET STEP ==================== //
    show_image(crop1, "crop1", 500); show_image(crop2, "crop2", 500);
    int isOne = verify(mobilefacenet, crop1, crop2, 0.4);

    if (isOne){
        printf("Same\n");
    } else {
        printf("Different\n");
    }
    
    free_image(im1); free_image(im2);
    free_image(crop1); free_image(crop2);
    free(dets);

    return 0;
}
#else
#include <stdio.h>
#include "parser.h"

int main(int argc, char** argv)
{
    image im = load_image_color("images/patch_112x96.jpg", 0, 0);   // RGB, 0.~1.
    image cvt = convert_mobilefacenet_image(im);                    // BGR, -1~1

    network* net = load_mobilefacenet();
    network_predict(net, cvt.data);
    
    FILE* fp = fopen("images/patch_112x96_c.txt", "w");
#if 0
    for (int i = 0; i < cvt.c*cvt.h*cvt.w; i++ ){
        fprintf(fp, "%.8f\n", cvt.data[i]);
    }
#else
    for (int i = 0; i < net->n; i++ ){
        if (i == 2){
            layer l = net->layers[i];
            for (int j = 0; j < l.outputs; j++ ){
                fprintf(fp, "%.8f\n", l.output[j]);
            }
        }
    }
#endif
    fclose(fp);
    free_image(im);
    free_image(cvt);
    return 0;
}

#endif