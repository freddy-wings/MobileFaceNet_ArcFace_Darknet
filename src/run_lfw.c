#include <stdio.h>
#include "darknet.h"
#include "utils.h"
#include "list.h"

#include "mobilefacenet.h"
#include "crop_align.h"


#define LFW_PATH "data/lfw"

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
        char* ptr = pair; char* s; int n = 0;
        list* sample = make_list();
        while((s = strtok(ptr, "\t")) != NULL){ // 以`Tab(\t)`分割字符串

            if (n<2) {                          // 前两个字符串,依次保存
                list_insert(sample, (void*)s);
            } else if (n==2) {                  // 第三个字符串
                if (strlen(s)>2) {              // 即不为数字,为人名,负样本
                    list_insert(sample, (void*)s);
                    list_insert(sample, (void*)strtok(ptr, "\t"));
                } else {                        // 为数字,正样本
                    list_insert(sample, (void*)sample->front->val);
                    list_insert(sample, (void*)s);
                }
            }
            n++; ptr = NULL;
        }
        list_insert(pairs, (void*)sample);
    }
    fclose(fp);
    return pairs;
}

int verify_lfw_images(int argc, char** argv)
{
    // ======================== INITIALIZE ======================== //
    params p = initParams(argc, argv);
    network* pnet = load_mtcnn_net("PNet");
    network* rnet = load_mtcnn_net("RNet");
    network* onet = load_mtcnn_net("ONet");
    landmark aligned = initAligned();
    network* mobilefacenet = load_mobilefacenet();
    printf("\033[2J"); printf("\033[1;1H");

    // ========================= LODA DATA ======================== //
    printf("Reading LFW...");
    list* pairs = get_lfw_pairs();
    printf("OK\n");

    // ======================= VERIFICATION ======================= //
    int size = pairs->size;
    double time = what_time_is_it_now();
    double elapsed = 0;
    for (int i = 0; i < pairs->size; i++ ){
        
        double duration = what_time_is_it_now() - time; 
        elapsed += duration; time = what_time_is_it_now();
        printf("\033[2J"); printf("\033[1;1H");
        printf("Pair [%4d]/[%4d] >> Elapsed %6.1fs\n",
                        i+1, size, elapsed);

        if (i == 494){
            // TODO
        }
        // --------------- GET A PAIR OF SAMPLE ------------------- //
        list* sample = (list*)list_pop(pairs);

        int idx2 = atoi((char*)list_pop(sample));
        char* name2 = (char*)list_pop(sample);
        int idx1 = atoi((char*)list_pop(sample));
        char* name1 = (char*)list_pop(sample);
        char buff[256];
        sprintf(buff, "%s/%s/%s_%04d.jpg", LFW_PATH, name1, name1, idx1);
        image image1 = load_image_color(buff, 0, 0);
        sprintf(buff, "%s/%s/%s_%04d.jpg", LFW_PATH, name2, name2, idx2);
        image image2 = load_image_color(buff, 0, 0);
        
        // show_image(image1, "image1", 100); show_image(image2, "image2", 0);

        // if (0 == strcmp(name1, name2))
    }
    return 0;
}