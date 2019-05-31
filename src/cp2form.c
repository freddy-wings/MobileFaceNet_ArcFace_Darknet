/*
 * @Author: louishsu 
 * @Date: 2019-05-31 13:53:34 
 * @Last Modified by: mikey.zhaopeng
 * @Last Modified time: 2019-05-31 17:12:34
 */
#include "cp2form.h"

/* 
 * @param
 *      xy: [x, y]， Nx2
 * @return
 *      ret: 2Nx4
 * @notes
 *      x  y 1 0
 *      y -x 0 1
 */
CvMat* _stitch(const CvMat* xy)
{
    int rows = xy->rows;
    CvMat* C;

    // x, y, 1, 0
    cvGetCol(xy, C, 0); CvMat* x = cvCloneMat(C);
    cvGetCol(xy, C, 0); CvMat* y = cvCloneMat(C);
    CvMat* ones  = cvCreateMat(rows, 1, CV_32FC1);
    CvMat* zeros = cvCreateMat(rows, 1, CV_32FC1);
    for (int i = 0; i < rows; i++){
        ones ->data.fl[i] = 1.; zeros->data.fl[i] = 0.;
    }
    
    CvMat* X = cvCreateMat(2*xy->rows, 4, CV_32FC1);
    cvGetSubRect(X, C, cvRect(0,    0, 1, rows)); cvCopy(x, C, NULL);
    cvGetSubRect(X, C, cvRect(1,    0, 1, rows)); cvCopy(y, C, NULL);
    cvGetSubRect(X, C, cvRect(2,    0, 1, rows)); cvCopy(ones,  C, NULL);
    cvGetSubRect(X, C, cvRect(3,    0, 1, rows)); cvCopy(zeros, C, NULL);
    for (int i = 0; i < rows; i++) x->data.fl[i] *= -1.;
    cvGetSubRect(X, C, cvRect(0, rows, 1, rows)); cvCopy(y, C, NULL);
    cvGetSubRect(X, C, cvRect(1, rows, 1, rows)); cvCopy(x, C, NULL);
    cvGetSubRect(X, C, cvRect(2, rows, 1, rows)); cvCopy(zeros, C, NULL);
    cvGetSubRect(X, C, cvRect(3, rows, 1, rows)); cvCopy(ones,  C, NULL);

    cvReleaseMat(&x); cvReleaseMat(&y);
    cvReleaseMat(&ones); cvReleaseMat(&zeros);

    return X;   
}


/* 
 * @param
 *      M:  2x3
 *      uv: [u, v]， Nx2
 * @return
 *      xy: Nx2
 * @notes
 *      xy = [uv, 1] * M^T, Nx2
 */
CvMat* _tformfwd(const CvMat* M, const CvMat* uv)
{
    int rows = uv->rows;
    int cols = uv->cols;
    CvMat* mat;

    CvMat* UV = cvCreateMat(rows, cols + 1, CV_32FC1);
    cvGetSubRect(UV, mat, cvRect(0, 0, cols, rows));
    cvCopy(uv, mat, NULL);
    for (int r = 0; r < rows; r++){
        UV->data.fl[r*(cols+1) + cols] = 1.;
    }

    CvMat* MT; CvMat* xy;
    cvTranspose(M, MT);
    cvMatMul(UV, MT, xy);

    cvReleaseMat(&UV);
    return xy;
}

/* 
 * @param
 *      uv: [u, v]， Nx2
 *      xy: [x, y]， Nx2
 * @return
 * @notes
 * -    Xr = Y   ===>  r = (X^T X + \lambda I)^{-1} X^T Y
 */
CvMat* _findNonreflectiveSimilarity(const CvMat* uv, const CvMat* xy)
{
    CvMat* X = _stitch(xy);                         // 2N x  4
    CvMat* XT;  cvTranspose(X, XT);                 //  4 x 2N
    CvMat* XTX; cvMatMul(XT, X, XTX);               //  4 x  4
    for (int i = 0; i < XTX->rows; i++) XTX->data.fl[i*XTX->rows + i] += 1e-15;
    CvMat* XTXi; cvInvert(XTX, XTXi, CV_LU);        //  4 x  4

    CvMat* Y; cvTranspose(uv, Y);                   //  2 x  N
    CvMat header; 
    Y = cvReshape(Y, &header, 0, 1);                //  1 x 2N
    cvTranspose(Y, Y);                              // 2N x  1
    
    CvMat* r;
    cvMatMul(XTXi, XT, r); cvMatMul(r, Y, r);       //  4 x  1
    cvReleaseMat(&X); 

    // -----------------------------------------------------------------------

    CvMat* R = cvCreateMat(3, 3, CV_32FC1);
    R->data.fl[0 * 3 + 0] = r->data.fl[0]; R->data.fl[0 * 3 + 1] = -r->data.fl[1]; R->data.fl[0 * 3 + 2] = 0.;
    R->data.fl[1 * 3 + 0] = r->data.fl[1]; R->data.fl[1 * 3 + 1] =  r->data.fl[0]; R->data.fl[1 * 3 + 2] = 0.;
    R->data.fl[2 * 3 + 0] = r->data.fl[2]; R->data.fl[2 * 3 + 1] =  r->data.fl[3]; R->data.fl[2 * 3 + 2] = 1.;
    CvMat* Ri; cvInvert(R, Ri, CV_LU);
    CvMat* MT; cvGetSubRect(Ri, MT, cvRect(0, 0, 2, 3));
    CvMat* M; cvTranspose(MT, M);
    cvReleaseMat(&R);

    return M;
}

/* 
 * @param
 *      uv: [u, v]， Nx2
 *      xy: [x, y]， Nx2
 * @return
 * @notes
 */
CvMat* _findReflectiveSimilarity(const CvMat* uv, const CvMat* xy)
{
    CvMat* xyR = cvCloneMat(xy);
    CvMat* C; 
    
    cvGetCol(xy, C, 0);
    for (int r = 0; r < C->rows; r++) C->data.fl[r] *= -1;
    
    CvMat* M1 = _findNonreflectiveSimilarity(uv, xy);
    CvMat* M2 = _findNonreflectiveSimilarity(uv, xyR);

    cvGetCol(M2, C, 0);
    for (int r = 0; r < C->rows; r++) C->data.fl[r] *= -1;

    CvMat* xy1 = _tformfwd(M1, uv);
    CvMat* xy2 = _tformfwd(M2, uv);
    double norm1 = cvNorm(xy1, xy, CV_L2, NULL);
    double norm2 = cvNorm(xy2, xy, CV_L2, NULL);

    if (norm1 > norm2)
        return M1;
    else
        return M2;
}

/* 
 * @param
 *      src: 原始坐标点 [[x1, y1], [x2, y2], ..., [xn, yn]]
 *      dst: 对齐对标点 [[x1, y1], [x2, y2], ..., [xn, yn]]
 *      mode:模式 
 * @return
 * @notes
 */
CvMat* cp2form(const CvMat* src, const CvMat* dst, int mode)
{
    CvMat* M;

    if (mode == 0){
        M = _findNonreflectiveSimilarity(src, dst);
    } else if (mode == 1){
        M = _findReflectiveSimilarity(src, dst);
    } else {
        printf("Mode %d not supported!\n", mode);
    }

    return M;
}