#include "mtcnn.h"
#include "mobilefacenet.h"

#if 1

int main(int argc, char** argv)
{
    if (find_arg(argc, argv, "--lfw")){
        verify_lfw_images(argc, argv);
    } else {
        verify_input_images(argc, argv);
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