#include "mtcnn.h"
#include "mobilefacenet.h"

#if 1

int main(int argc, char** argv)
{
    int help = find_arg(argc, argv, "--help");
    if(help){
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "    ./mobilefacenet <function>\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Optional:\n");
        fprintf(stderr, "    --video    video mode;\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "  for MobileFacenet:\n");
        fprintf(stderr, "    --image1   path of image1  default `images/Aaron_Peirsol_0001.jpg`;\n");
        fprintf(stderr, "    --image2   path of image2  default `images/Aaron_Peirsol_0002.jpg`;\n");
        fprintf(stderr, "    --mode     align mode      default `1`, find similarity;\n");
        fprintf(stderr, "    --dataset  eval dataset    default `NULL`;\n");
        fprintf(stderr, "    --aligned  aligned images  default `0`;\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "  for MTCNN:\n");
        fprintf(stderr, "    -v         video mode,     default `0`, image mode;\n");
        fprintf(stderr, "    --path     file path,      default `../images/test.*`;\n");
        fprintf(stderr, "    --index    camera index,   default `0`;\n");
        fprintf(stderr, "    -p         thresh for PNet,default `0.8`;\n");
        fprintf(stderr, "    -r         thresh for RNet,defalut `0.8`;\n");
        fprintf(stderr, "    -o         thresh for ONet,defalut `0.8`;\n");
        fprintf(stderr, "    --minface  minimal face,   default `96.0`;\n");
        fprintf(stderr, "    --scale    resize factor,  default `0.79`;\n");
        fprintf(stderr, "    --stride                   default `2`;\n");
        fprintf(stderr, "    --cellsize                 default `12`;\n");
        fprintf(stderr, "    --softnms                  default `0`;\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Example:\n");
        fprintf(stderr, "    ./mobilefacenet\n");
        fprintf(stderr, "    ./mobilefacenet --image1 [path1] --image2 [path2]\n");
        fprintf(stderr, "    ./mobilefacenet --dataset lfw --minface 36\n");
        fprintf(stderr, "    ./mobilefacenet --dataset lfw  --aligned\n");
        fprintf(stderr, "    ./mobilefacenet --video\n");
        fprintf(stderr, "\n");
        return 0;
    }

    int video = find_arg(argc, argv, "--video");
    char* dataset = find_char_arg(argc, argv, "--dataset", NULL);

    if (video){
        verify_video_demo(argc, argv);
        return 0;
    }

    if (!dataset){
        verify_input_images(argc, argv);
    } else if (0 == strcmp(dataset, "lfw")){
        verify_lfw_images(argc, argv);
    }

    return 0;
}

#endif

#if 0
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
        if ( i == 0 || i == 1 ||i == 2 || i == 5 || i == 13 || i == 22 || i == 31 || i == 40 || i == 49 ||
             i == 57 || i == 66 || i == 75 || i == 84 || i == 93 || i == 102 || i == 111 ||
              i == 119 || i == 128 || i == 137 || i == 140 || i == 141 || i == 142 || i == 143 || i == 144){
            layer l = net->layers[i];
            fprintf(fp, "[%d]%s =================\n", i, layer_type_to_string(l.type));
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