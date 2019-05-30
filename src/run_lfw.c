#include <stdio.h>
 #include <unistd.h>
#include "darknet.h"
#include "utils.h"
#include "list.h"

#include "mobilefacenet.h"
#include "crop_align.h"

#define LFW_PATH "data/lfw"
#define LFW_ALIGNED_PATH "data/lfw-112X96"  // download from [Xiaoccer/MobileFaceNet_Pytorch/README.md/Usage/](https://github.com/Xiaoccer/MobileFaceNet_Pytorch)

list* get_lfw_pairs()
{
    char* pair;
    char* path = "./data/pairs.txt";
    FILE* fp = fopen(path, "r");
    if(!fp) file_error(path);
    fgetl(fp);                                  // 第一行文本不需要
    
    list* pairs = make_list();
    while ((pair = fgetl(fp))){                 // 获取一行文本
        /* Sample example:
         *      positive: Abel_Pacheco	1	4
         *      negative: Abdel_Madi_Shabneh	1	Dean_Barker	1
        */
        char* ptr = pair; char* s;
        list* sample = make_list();
        while((s = strtok(ptr, "\t")) != NULL){ // 以`Tab(\t)`分割字符串
            list_insert(sample, (void*)s);
            ptr = NULL;
        }

        list_insert(pairs, (void*)sample);
    }
    fclose(fp);
    return pairs;
}

int verify_lfw_images(int argc, char** argv)
{
    // ======================== INITIALIZE ======================== //
    float thresh = find_float_arg(argc, argv, "--thresh", 0.3);
    int align = find_arg(argc, argv, "--aligned");

    FILE* fp_bad = fopen("log/bad_samples.txt", "w");
    FILE* fp_score = fopen("log/cosine_score.txt", "w");

    params p = initParams(argc, argv);
    network* pnet = load_mtcnn_net("PNet");
    network* rnet = load_mtcnn_net("RNet");
    network* onet = load_mtcnn_net("ONet");

    landmark aligned = initAligned();
    network* mobilefacenet = load_mobilefacenet();
    printf("\033[2J"); printf("\033[1;1H");

    // ========================= LODA DATA ======================== //
    printf("Reading LFW...");
    int bad_samples = 0;
    int tp=0, fp=0, tn=0, fn=0;
    list* pairs = get_lfw_pairs();
    printf("OK\n");

    // ======================= VERIFICATION ======================= //
    double time = what_time_is_it_now();
    double elapsed = 0;
    int total_samples = pairs->size;
    for (int i = 0; i < total_samples; i++ ){
        
        double duration = what_time_is_it_now() - time; 
        elapsed += duration; time = what_time_is_it_now();
        printf("\033[2;1H"); printf("\033[K"); 
        printf("[%4d]/[%4d] >> Elapsed %6.1fs, FPS %.2f\n",
                        i+1, total_samples, elapsed, 1. / duration);

        // --------------- GET A PAIR OF SAMPLE ------------------- //
        list* sample = (list*)list_pop(pairs);
        char* name1 = NULL; char* name2 = NULL; 
        int idx1 = -1, idx2 = -1;
        if (sample->size == 3){
            idx2 = atoi((char*)list_pop(sample));
            idx1 = atoi((char*)list_pop(sample));
            name2 = (char*)list_pop(sample);
            name1 = (char*)name2;
        } else {
            idx2 = atoi((char*)list_pop(sample));
            name2 = (char*)list_pop(sample);
            idx1 = atoi((char*)list_pop(sample));
            name1 = (char*)list_pop(sample);
        }
        
        char buff[256];
        if (!align)
            sprintf(buff, "%s/%s/%s_%04d.jpg", LFW_PATH, name1, name1, idx1);
        else 
            sprintf(buff, "%s/%s/%s_%04d.jpg", LFW_ALIGNED_PATH, name1, name1, idx1);
        image im1 = load_image_color(buff, 0, 0);

        if (!align)
            sprintf(buff, "%s/%s/%s_%04d.jpg", LFW_PATH, name2, name2, idx2);
        else 
            sprintf(buff, "%s/%s/%s_%04d.jpg", LFW_ALIGNED_PATH, name2, name2, idx2);
        image im2 = load_image_color(buff, 0, 0);

        // show_image(im1, "im1", 10); show_image(im2, "im2", 0);

        int pred = -1;
        float cosine = thresh;
        
        if (!align){
            // ------------------- DETECT FACE ------------------------ //
            int idx = 0;
            // detect im1
            detect* dets = calloc(0, sizeof(dets)); int n = 0;
            detect_image(pnet, rnet, onet, im1, &n, &dets, p);
            if (n == 0){
                bad_samples += 1; 
                fprintf(fp_bad, "detected error: %d\n", i);
                printf("Image1 is not detected!\n"); 
                continue;
            }
            // show_detect(im1, dets, n, "detect1", 0, 1, 1, 1);
            idx = keep_one(dets, n, im1); 
            bbox box1 = dets[idx].bx; landmark landmark1 = dets[idx].mk;
            landmark1 = substract_bias(landmark1, box1.x1, box1.y1);

            // detect im2
            dets = realloc(dets, 0); n = 0;
            detect_image(pnet, rnet, onet, im2, &n, &dets, p);
            if (n == 0){
                bad_samples += 1; 
                fprintf(fp_bad, "detected error: %d\n", i);
                printf("Image2 is not detected!\n"); 
                continue;
            }
            // show_detect(im2, dets, n, "detect2", 0, 1, 1, 1);
            idx = keep_one(dets, n, im2); 
            bbox box2 = dets[idx].bx; landmark landmark2 = dets[idx].mk;
            landmark2 = substract_bias(landmark2, box2.x1, box2.y1);

            // ------------------- VERIFICATION ----------------------- //
            image crop1 = crop_image_by_box(im1, box1, H, W);
            image crop2 = crop_image_by_box(im2, box2, H, W);
            image warped1 = align_image_with_landmark(crop1, landmark1, aligned);
            image warped2 = align_image_with_landmark(crop2, landmark2, aligned);
            // show_image(warped1, "warped1", 10); show_image(warped2, "warped2", 0);

            pred = verify(mobilefacenet, warped1, warped2, &cosine);// if matched, pred = 1, else 0
            free_image(crop1); free_image(crop2);
            free_image(warped1); free_image(warped2);
            free(dets);
        } else {
            pred = verify(mobilefacenet, im1, im2, &cosine);        // if matched, pred = 1, else 0
        }
        free_image(im1); free_image(im2); 


        int gt = (0 == strcmp(name1, name2));                       // if mathced, gt = 1, else 0
        if (gt == 1){
            if (pred == 1)
                tp += 1;    // gt = 1, pred = 1, true positive
            else
                fn += 1 ;   // gt = 1, pred = 0, false negative
        } else {
            if (pred == 1)
                fp += 1;    // gt = 0, pred = 1, false positive
            else
                tn += 1;    // gt = 0, pred = 0, true negative
        }

        fprintf(fp_score, "%d %d %.6f\n", i, gt, cosine);
        if (gt != pred) {
            fprintf(fp_bad, "verify error: %d, gt: %d, pred: %d, dist=%3.2f\n", i, gt, pred, cosine);
        }
        printf("Gt: %d  Pred: %d  Cosine: %3.2f\n", gt, pred, cosine);
    }
    
    // ========================= STATISTIC ======================== //
    printf("\033[4;1H"); printf("\033[K"); 

    int detected_samples = total_samples - bad_samples;
    float accuracy = (float)(tp + tn) / (float)detected_samples;
    float precision = (float)(tp) / (float)(tp + fp);
    float recall = (float)(tp) / (float)(tp + fn);
    
    printf("Total: %4d | Bad: %4d | Detected: %4d\n", 
            total_samples, bad_samples, detected_samples);
    printf("Accuracy: %2.2f | Precision: %2.2f | Recall: %2.2f\n", 
            accuracy, precision, recall);

    fclose(fp_bad); fclose(fp_score);
    free(pairs);

    return 0;
}
