#include "mobilefacenet.h"
#include "crop_align.h"

int verify_input_images(int argc, char** argv)
{
    // ======================== INITIALIZE ======================== //
    params p = initParams(argc, argv);
    network* pnet = load_mtcnn_net("PNet");
    network* rnet = load_mtcnn_net("RNet");
    network* onet = load_mtcnn_net("ONet");
    landmark alignOffset = initAlignedOffset();
    network* mobilefacenet = load_mobilefacenet();
    printf("\033[2J"); printf("\033[1;1H");

    // ======================= LOAD IMAGES ======================== //
    printf("Loading images...");
    // char* imgpath1 = find_char_arg(argc, argv, "--image1", "images/Aaron_Eckhart_0001.jpg");
    char* imgpath1 = find_char_arg(argc, argv, "--image1", "images/Aaron_Peirsol_0001.jpg");
    char* imgpath2 = find_char_arg(argc, argv, "--image2", "images/Aaron_Peirsol_0002.jpg");

    if (0 == strcmp(imgpath1, "images/Aaron_Peirsol_0001.jpg"))
        printf("\nUsing default image 1: %s\n", imgpath1);
    if (0 == strcmp(imgpath2, "images/Aaron_Peirsol_0002.jpg"))
        printf("Using default image 2: %s\n", imgpath2);

    image im1 = load_image_color(imgpath1, 0, 0); 
    // im1 = rotate_image(im1, 0.4);
    image im2 = load_image_color(imgpath2, 0, 0); 
    show_image(im1, "im1", 1000); show_image(im2, "im2", 1000);
    printf("OK\n");
    
    // ======================== MTCNN STEP ======================== //
    printf("Detecting face...");
    int idx = 0;
    // detect image1
    detect* dets = calloc(0, sizeof(dets)); int n = 0;
    detect_image(pnet, rnet, onet, im1, &n, &dets, p);
    show_detect(im1, dets, n, "detect1", 1000, 1, 1, 1);
    if (n == 0){printf("\nimage 1 is not detected!\n"); return -1;}
    idx = keep_one(dets, n, im1); 
    bbox box1 = dets[idx].bx; landmark landmark1 = dets[idx].mk;

    // detect image2
    dets = realloc(dets, 0); n = 0;
    detect_image(pnet, rnet, onet, im2, &n, &dets, p);
    show_detect(im2, dets, n, "detect2", 1000, 1, 1, 1);
    if (n == 0){printf("\nimage 2 is not detected!\n"); return -1;}
    idx = keep_one(dets, n, im2); 
    bbox box2 = dets[idx].bx; landmark landmark2 = dets[idx].mk;
    printf("OK\n");

    // ==================== MOBILEFACENET STEP ==================== //
    printf("Croping face...");
    image warped1 = image_crop_aligned(im1, box1, landmark1, alignOffset, H, W);
    image warped2 = image_crop_aligned(im2, box2, landmark2, alignOffset, H, W);
    show_image(warped1, "warped1", 1000); show_image(warped2, "warped2", 1000);
    printf("OK\n");

    printf("Verifying...");
    float cosine = find_float_arg(argc, argv, "--thresh", 0.3);
    int is_one = verify(mobilefacenet, warped1, warped2, &cosine);
    printf("OK\n");

    printf("\nCosine=%3.2f >>> ", cosine);
    if (is_one){
        printf("Same\n");
    } else {
        printf("Different\n");
    }
    
    free_image(im1); free_image(im2);
    free_image(warped1); free_image(warped2);
    free(dets);

    return is_one;
}