#include "mobilefacenet.h"
#include "crop_align.h"

int verify_input_images(int argc, char** argv)
{
    // ======================== INITIALIZE ======================== //
    params p = initParams(argc, argv);
    network* pnet = load_mtcnn_net("PNet");
    network* rnet = load_mtcnn_net("RNet");
    network* onet = load_mtcnn_net("ONet");
    landmark aligned = initAligned();
    network* mobilefacenet = load_mobilefacenet();
    printf("\033[2J"); printf("\033[1;1H");

    // ======================= LOAD IMAGES ======================== //
    printf("Loading images...");
    // char* imgpath1 = find_char_arg(argc, argv, "--image1", "images/Aaron_Eckhart_0001.jpg");
    char* imgpath1 = find_char_arg(argc, argv, "--image1", "images/Aaron_Peirsol_0001.jpg");
    char* imgpath2 = find_char_arg(argc, argv, "--image2", "images/Aaron_Peirsol_0002.jpg");
    image im1 = load_image_color(imgpath1, 0, 0); 
    // im1 = rotate_image(im1, 0.4);
    image im2 = load_image_color(imgpath2, 0, 0); 
    show_image(im1, "im1", 500); show_image(im2, "im2", 500);
    printf("OK\n");
    
    // ======================== MTCNN STEP ======================== //
    printf("Detecting face...");
    int idx = 0;
    // detect image1
    detect* dets = calloc(0, sizeof(dets)); int n = 0;
    detect_image(pnet, rnet, onet, im1, &n, &dets, p);
    // show_detect(im1, dets, n, "detect1", 500, 1, 1, 1);
    if (n == 0){printf("image 1 is not detected!\n"); return -1;}
    idx = keep_one(dets, n, im1); 
    bbox box1 = dets[idx].bx; landmark landmark1 = dets[idx].mk;
    landmark1 = substract_bias(landmark1, box1.x1, box1.y1);

    // detect image2
    dets = realloc(dets, 0); n = 0;
    detect_image(pnet, rnet, onet, im2, &n, &dets, p);
    // show_detect(im2, dets, n, "detect2", 500, 1, 1, 1);
    if (n == 0){printf("image 2 is not detected!\n"); return -1;}
    idx = keep_one(dets, n, im2); 
    bbox box2 = dets[idx].bx; landmark landmark2 = dets[idx].mk;
    landmark2 = substract_bias(landmark2, box2.x1, box2.y1);
    printf("OK\n");

    // ==================== MOBILEFACENET STEP ==================== //
    printf("Croping face...");
    image crop1 = crop_image_by_box(im1, box1, H, W);
    image crop2 = crop_image_by_box(im2, box2, H, W);
    image warped1 = align_image_with_landmark(crop1, landmark1, aligned);
    image warped2 = align_image_with_landmark(crop2, landmark2, aligned);
    // show_image(crop1, "crop1", 500); show_image(crop2, "crop2", 500);
    show_image(warped1, "warped1", 500); show_image(warped2, "warped2", 500);
    printf("OK\n");

    printf("Verifying...");
    float cosine = find_float_arg(argc, argv, "--thresh", 0.3);
    int is_one = verify(mobilefacenet, crop1, crop2, &cosine);
    printf("OK\n");

    printf("Cosine=%3.2f >>> ", cosine);
    if (is_one){
        printf("Same\n");
    } else {
        printf("Different\n");
    }
    
    free_image(im1); free_image(im2);
    free_image(crop1); free_image(crop2);
    free_image(warped1); free_image(warped2);
    free(dets);

    return is_one;
}