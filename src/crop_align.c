#include "crop_align.h"


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

landmark substract_bias(landmark mark, float x, float y)
{
    mark.x1 -= x; mark.y1 -= y;
    mark.x2 -= x; mark.y2 -= y;
    mark.x3 -= x; mark.y3 -= y;
    mark.x4 -= x; mark.y4 -= y;
    mark.x5 -= x; mark.y5 -= y;
    return mark;
}

image align_image_with_landmark(image im, landmark src, landmark dst)
{
    CvPoint2D32f srcPts[5], dstPts[5];
    
    srcPts[0].x = src.x1;
    srcPts[0].y = src.y1;
    srcPts[1].x = src.x2;
    srcPts[1].y = src.y2;
    srcPts[2].x = src.x3;
    srcPts[2].y = src.y3;
    srcPts[3].x = src.x4;
    srcPts[3].y = src.y4;
    srcPts[4].x = src.x5;
    srcPts[4].y = src.y5;
    
    dstPts[0].x = dst.x1;
    dstPts[0].y = dst.y1;
    dstPts[1].x = dst.x2;
    dstPts[1].y = dst.y2;
    dstPts[2].x = dst.x3;
    dstPts[2].y = dst.y3;
    dstPts[3].x = dst.x4;
    dstPts[3].y = dst.y4;
    dstPts[4].x = dst.x5;
    dstPts[4].y = dst.y5;

    CvMat* warpMat = cvCreateMat(2, 3, CV_32FC1);
    cvGetAffineTransform(srcPts, dstPts, warpMat);
    
    IplImage* srcIpl = image_to_ipl(im);
    IplImage* dstIpl = cvCloneImage(srcIpl);
    cvWarpAffine(srcIpl, dstIpl, warpMat, CV_INTER_LINEAR+CV_WARP_INVERSE_MAP, cvScalarAll(0));

    image warped = ipl_to_image(dstIpl);
    return warped;
}
