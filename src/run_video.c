#include <unistd.h>
#include "mtcnn.h"
#include "g_mobilefacenet.h"
#include "crop_align.h"

// 显示相关
static int g_videoDone = 0;
static char* g_winname = "frame";
static char* g_trackbarDetection    = "detect";
static char* g_trackbarVerification = "verify";
static int g_valueMax = 100;
static int g_valueDetection = 0;
static int g_valueVerification = 0;
static CvFont g_font;

// 图像采集
static CvCapture* g_cvCap = NULL;
static image g_imFrame[3];
static int g_index = 0;

static double g_time;
static double g_fps;
static int g_running = 0;

// 不晃
static int g_noFrame = 0;               // 无人脸帧计数，当超过5帧无人脸，则清除`g_box`等
static float g_filter = 0.85;           // 滤波系数
static float g_score = 0;
static bbox g_box = {0};
static landmark g_mark = {0};
static landmark g_aligned = {0};

// 检测
static params g_mtcnnParam;             // MTCNN参数
static network* g_pnet; 
static network* g_rnet;
static network* g_onet;
static detect* g_dets = NULL;           // 当前帧检测得结果
static int g_ndets = 0;                 // 当前帧检测人脸个数

// 验证
#define N 128                           // 单个特征维度
static network* g_mobilefacenet;
static int g_mode = 0;                  // 图像对齐模式
static int g_initialized = 0;           // 已保存有特征
static float* g_feat_saved = NULL;      // 已保存特征
static float* g_feat_toverify = NULL;   // 待验证特征
static float g_cosine = 0;              // 计算得余弦值
static float g_thresh = 0.5;            // 阈值
static int g_isOne = -1;                // 是否为同一人
static char g_featurefile[256] = {0};   // 保存文件名

/*
 * 读取一帧图像
 * @params:
 * @returns:
 * -    dst: RGB
 */
image _frame()
{
    IplImage* iplFrame = cvQueryFrame(g_cvCap);
    image dst = ipl_to_image(iplFrame); // BGR
    rgbgr_image(dst);                   // RGB
    return dst;
}

/*
 * 多线程读取图像函数，存放在当前索引位置
 * @params:
 * -    ptr
 */
void* read_frame_in_thread(void* ptr)
{
    free_image(g_imFrame[g_index]);
    g_imFrame[g_index] = _frame();
    if (g_imFrame[g_index].data == 0){
        g_videoDone = 1;
        return 0;
    }
    return 0;
}

/*
 * 多线程检测图像函数
 * @params:
 * -    ptr
 */
void* detect_frame_in_thread(void* ptr)
{
    g_running = 1;

    image frame = g_imFrame[(g_index + 2) % 3];
    g_dets = realloc(g_dets, 0); g_ndets = 0;
    detect_image(g_pnet, g_rnet, g_onet, frame, &g_ndets, &g_dets, g_mtcnnParam);

    g_running = 0;
}

/*
 * 计算人脸特征，输出256维
 * @params:
 * -    im: 输入RGB图像
 * -    mark: 检测到的人脸关键点位置
 * -    X:  特征地址
 */
void generate_feature(image im, landmark mark, float* X)
{
    float* x = NULL;
    image warped = image_aligned_v2(im, mark, g_aligned, H, W, g_mode);
    image cvt = convert_mobilefacenet_image(warped);
    
    x = network_predict(g_mobilefacenet, cvt.data);
    memcpy(X,     x, N*sizeof(float));

    flip_image(cvt);
    x = network_predict(g_mobilefacenet, cvt.data);
    memcpy(X + N, x, N*sizeof(float));

    free_image(warped); free_image(cvt);
}

/*
 * 保存人脸特征至文件
 * @params:
 * -    filename: 保存文件名
 * -    X:  特征地址
 */
void save_feature(char* filename, float* X)
{
    FILE* fp = fopen(filename, "w");
    fwrite(X, sizeof(float), N*2, fp);
    fclose(fp);
}

/*
 * 读取人脸特征至文件
 * @params:
 * -    filename: 读取文件名
 * -    X:  特征地址
 */
void load_feature(char* filename, float* X)
{
    FILE* fp = fopen(filename, "r");
    fread(X, sizeof(float), N*2, fp);
    fclose(fp);
}

#if 0
/*
 * 2019.7.2 删除 
 */
void* display_frame_in_thread(void* ptr)
{
    while(g_running);

    image im = g_imFrame[(g_index + 1) % 3];
    IplImage* iplFrame = image_to_ipl(im);
    for (int i = 0; i < g_ndets; i++ ){
        detect det = g_dets[i];
        float score = det.score;
        bbox bx = det.bx;
        landmark mk = det.mk;

        char buff[256];
        sprintf(buff, "%.2f", score);
        cvPutText(iplFrame, buff, cvPoint((int)bx.x1, (int)bx.y1),
                    &font, cvScalar(0, 0, 255, 0));

        cvRectangle(iplFrame, cvPoint((int)bx.x1, (int)bx.y1),
                    cvPoint((int)bx.x2, (int)bx.y2),
                    cvScalar(255, 255, 255, 0), 1, 8, 0);

        cvCircle(iplFrame, cvPoint((int)mk.x1, (int)mk.y1),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x2, (int)mk.y2),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x3, (int)mk.y3),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x4, (int)mk.y4),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        cvCircle(iplFrame, cvPoint((int)mk.x5, (int)mk.y5),
                    1, cvScalar(255, 255, 255, 0), 1, 8, 0);
    }
    im = ipl_to_image(iplFrame);
    cvReleaseImage(&iplFrame);

    int c = show_image(im, g_winname, 1);


    if (c != -1) c = c%256;
    if (c == 27) {          // Esc
        g_videoDone = 1;
        return 0;
    } else if (c == 's') {  // save feature
        im = g_imFrame[(g_index + 1) % 3];
        int idx = keep_one(g_dets, g_ndets, im);
        if (idx < 0) return 0;
        bbox box = g_dets[idx].bx; landmark mark = g_dets[idx].mk;
        generate_feature(im, mark, g_feat_saved);
        g_initialized = 1;
    } else if (c == 'v') {  // verify
        im = g_imFrame[(g_index + 1) % 3];
        int idx = keep_one(g_dets, g_ndets, im);
        if (idx < 0) return 0;
        bbox box = g_dets[idx].bx; landmark mark = g_dets[idx].mk;
        generate_feature(im, mark, g_feat_toverify);

        g_cosine = distCosine(g_feat_saved, g_feat_toverify, N*2);
        g_isOne = g_cosine < g_thresh? 0: 1;
    } else if (c == '[') {
        g_thresh -= 0.05;
    } else if (c == ']') {
        g_thresh += 0.05;
    }

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n", g_fps);
    printf("Objects:%d\n\n", g_ndets);
    printf("Initialized:%d\n", g_initialized);
    printf("Thresh:%.4f\n", g_thresh);
    printf("Cosine:%.4f\n", g_cosine);
    printf("Verify:%d\n", g_isOne);

    return 0;
}

#else
/*
 * 多线程图像显示
 * 2019.7.2 添加
 * @params:
 * -    ptr
 * @notes:
 * -    为防止框抖动，增加滤波；
 * -    仅显示一个人脸框；
 */
void* display_frame_in_thread(void* ptr)
{
    while(g_running);

    image im = g_imFrame[(g_index + 1) % 3];
    IplImage* iplFrame = image_to_ipl(im);
    
    float filter_inv = 1 - g_filter;

    int i = keep_one(g_dets, g_ndets, im);
    if (i != -1){
        g_noFrame = 0;

        detect det = g_dets[i];
        g_score = det.score;
        g_valueDetection = (int)(g_score*100);
        if (g_box.x1 == 0 && g_box.x2 == 0 && g_box.y1 == 0 && g_box.y2 == 0){
            g_box = det.bx;
            g_mark = det.mk;
        } else {
            g_box.x1  = g_filter*g_box.x1 + filter_inv*det.bx.x1;
            g_box.x2  = g_filter*g_box.x2 + filter_inv*det.bx.x2;
            g_box.y1  = g_filter*g_box.y1 + filter_inv*det.bx.y1;
            g_box.y2  = g_filter*g_box.y2 + filter_inv*det.bx.y2;
            // g_mark.x1 = g_filter*g_mark.x1 + filter_inv*det.mk.x1;
            // g_mark.x2 = g_filter*g_mark.x2 + filter_inv*det.mk.x2;
            // g_mark.x3 = g_filter*g_mark.x3 + filter_inv*det.mk.x3;
            // g_mark.x4 = g_filter*g_mark.x4 + filter_inv*det.mk.x4;
            // g_mark.x5 = g_filter*g_mark.x5 + filter_inv*det.mk.x5;
            // g_mark.y1 = g_filter*g_mark.y1 + filter_inv*det.mk.y1;
            // g_mark.y2 = g_filter*g_mark.y2 + filter_inv*det.mk.y2;
            // g_mark.y3 = g_filter*g_mark.y3 + filter_inv*det.mk.y3;
            // g_mark.y4 = g_filter*g_mark.y4 + filter_inv*det.mk.y4;
            // g_mark.y5 = g_filter*g_mark.y5 + filter_inv*det.mk.y5;
        }

    } else {
        if (g_noFrame < 5){
            g_noFrame++;
        } else {
            g_cosine = 0; g_score = 0; g_box.x1  = 0; g_box.x2  = 0; g_box.y1  = 0; g_box.y2  = 0;
            // g_mark.x1 = 0; g_mark.x2 = 0; g_mark.x3 = 0; g_mark.x4 = 0; g_mark.x5 = 0;
            // g_mark.y1 = 0; g_mark.y2 = 0; g_mark.y3 = 0; g_mark.y4 = 0; g_mark.y5 = 0;
            g_valueDetection = g_valueVerification = 0;
        }
    }

    if (g_noFrame < 5){
        char buff[256]; 
        sprintf(buff, "%.2f", g_score);
        cvPutText(iplFrame, buff, cvPoint((int)g_box.x1, (int)g_box.y1),
                    &g_font, cvScalar(0, 0, 255, 0));
        sprintf(buff, "%.2f", g_cosine);
        cvPutText(iplFrame, buff, cvPoint((int)g_box.x2, (int)g_box.y1),
                    &g_font, cvScalar(0, 0, 255, 0));

        cvRectangle(iplFrame, cvPoint((int)g_box.x1, (int)g_box.y1),
                    cvPoint((int)g_box.x2, (int)g_box.y2),
                    cvScalar(255, 255, 255, 0), 1, 8, 0);

        cvSetTrackbarPos(g_trackbarDetection, g_winname, g_valueDetection);
        cvSetTrackbarPos(g_trackbarVerification, g_winname, g_valueVerification);

        // cvCircle(iplFrame, cvPoint((int)g_mark.x1, (int)g_mark.y1),
        //             1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        // cvCircle(iplFrame, cvPoint((int)g_mark.x2, (int)g_mark.y2),
        //             1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        // cvCircle(iplFrame, cvPoint((int)g_mark.x3, (int)g_mark.y3),
        //             1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        // cvCircle(iplFrame, cvPoint((int)g_mark.x4, (int)g_mark.y4),
        //             1, cvScalar(255, 255, 255, 0), 1, 8, 0);
        // cvCircle(iplFrame, cvPoint((int)g_mark.x5, (int)g_mark.y5),
        //             1, cvScalar(255, 255, 255, 0), 1, 8, 0);
    }

    im = ipl_to_image(iplFrame);
    cvReleaseImage(&iplFrame);

    int c = show_image(im, g_winname, 1);

    if (c != -1) c = c%256;
    if (c == 27) {          // Esc
        g_videoDone = 1;
        return 0;
    } else if (c == 's') {  // save feature
        im = g_imFrame[(g_index + 1) % 3];
        int idx = keep_one(g_dets, g_ndets, im);
        if (idx < 0) return 0;
        bbox box = g_dets[idx].bx; landmark mark = g_dets[idx].mk;
        generate_feature(im, mark, g_feat_saved);
        save_feature(g_featurefile, g_feat_saved);

        g_initialized = 1;
    } else if (c == 'v') {  // verify
        im = g_imFrame[(g_index + 1) % 3];
        int idx = keep_one(g_dets, g_ndets, im);
        if (idx < 0) return 0;
        bbox box = g_dets[idx].bx; landmark mark = g_dets[idx].mk;
        generate_feature(im, mark, g_feat_toverify);

        g_cosine = distCosine(g_feat_saved, g_feat_toverify, N*2);
        g_isOne = g_cosine < g_thresh? 0: 1;
        g_valueVerification = (int)(g_cosine*100);
    } else if (c == '[') {
        g_thresh -= 0.05;
    } else if (c == ']') {
        g_thresh += 0.05;
    }

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n", g_fps);
    printf("Objects:%d\n\n", g_ndets);
    printf("Initialized:%d\n", g_initialized);
    printf("Thresh:%.4f\n", g_thresh);
    printf("Cosine:%.4f\n", g_cosine);
    printf("Verify:%d\n", g_isOne);

    return 0;
}
#endif

/*
 * 视频demo主函数
 */
int verify_video_demo(int argc, char **argv)
{
    g_pnet = load_mtcnn_net("PNet");
    g_rnet = load_mtcnn_net("RNet");
    g_onet = load_mtcnn_net("ONet");
    g_mobilefacenet = load_mobilefacenet();
    printf("\n\n");

    // =================================================================================
    printf("Initializing Capture...");
    int index = find_int_arg(argc, argv, "--index", 0);
    if (index < 0){
        char* filepath = find_char_arg(argc, argv, "--path", "../images/test.mp4");
        if(0==strcmp(filepath, "../images/test.mp4")){
            fprintf(stderr, "Using default: %s\n", filepath);
        }
        g_cvCap = cvCaptureFromFile(filepath);
    } else {
        g_cvCap = cvCaptureFromCAM(index);
    }
    if (!g_cvCap){
        printf("failed!\n");
        return -1;
    }
    // cvSetCaptureProperty(g_cvCap, CV_CAP_PROP_FRAME_HEIGHT, H);
    // cvSetCaptureProperty(g_cvCap, CV_CAP_PROP_FRAME_WIDTH, W);
    
    g_imFrame[0] = _frame();
    g_imFrame[1] = copy_image(g_imFrame[0]);
    g_imFrame[2] = copy_image(g_imFrame[0]);

    cvNamedWindow(g_winname, CV_WINDOW_AUTOSIZE);
    cvCreateTrackbar(g_trackbarDetection, g_winname, &g_valueDetection, g_valueMax, NULL);
    cvCreateTrackbar(g_trackbarVerification, g_winname, &g_valueVerification, g_valueMax, NULL);
    cvInitFont(&g_font, CV_FONT_HERSHEY_SIMPLEX, 0.8, 0.8, 1, 2, 8);
    printf("OK!\n");

    // =================================================================================
    printf("Initializing detection...");
    g_mtcnnParam = initParams(argc, argv);
    g_dets = calloc(0, sizeof(detect)); g_ndets = 0;
    printf("OK!\n");

    // =================================================================================
    printf("Initializing verification...");
    // g_aligned = initAlignedOffset();
    g_aligned = initAligned();
    g_mode = find_int_arg(argc, argv, "--mode", 1);
    g_thresh = find_float_arg(argc, argv, "--thresh", 0.5);
    g_feat_saved = calloc(2*N, sizeof(float));
    g_feat_toverify = calloc(2*N, sizeof(float));
    char* name = find_char_arg(argc, argv, "--name", "xyb");
    sprintf(g_featurefile, "build/%s.feature", name);
    if (access(g_featurefile, F_OK) == 0){
        load_feature(g_featurefile, g_feat_saved);
        g_initialized = 1;
    }
    printf("OK!\n");

    // =================================================================================
    pthread_t thread_read;
    pthread_t thread_detect;
    pthread_t thread_display;

    // =================================================================================
    g_time = what_time_is_it_now();
    while(!g_videoDone){
        g_index = (g_index + 1) % 3;

        if(pthread_create(&thread_read, 0, read_frame_in_thread, 0)) error("Thread read create failed");
        if(pthread_create(&thread_detect, 0, detect_frame_in_thread, 0)) error("Thread detect create failed");
        if(pthread_create(&thread_display, 0, display_frame_in_thread, 0)) error("Thread detect create failed");
        
        g_fps = 1./(what_time_is_it_now() - g_time);
        g_time = what_time_is_it_now();
        // display_frame_in_thread(0);
        
        pthread_join(thread_read, 0);
        pthread_join(thread_detect, 0);
        pthread_join(thread_display, 0);
    }
    // =================================================================================
    for (int i = 0; i < 3; i++ ){
        free_image(g_imFrame[i]);
    }
    free(g_dets);
    cvReleaseCapture(&g_cvCap);
    cvDestroyWindow(g_winname);

    free_network(g_pnet);
    free_network(g_rnet);
    free_network(g_onet);
    free_network(g_mobilefacenet);
    
    free(g_feat_saved); free(g_feat_toverify);

    return 0;
}

