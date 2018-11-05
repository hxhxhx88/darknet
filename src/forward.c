#include "network.h"

void create_detector(int xpu, char *cfgfile, char *weightfile, network **net) {
#ifdef GPU
    if (xpu < 0) {
        // if compiled as GPU version, CPU is not supported, since this code sucks.
        fprintf(stderr, "GPU version darknet does not support running on CPU\n");
        exit(-1);
    }
#endif

    // gpu_index is a global variable
    gpu_index = xpu;

    // load the network
    *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(*net, 1);
}

void forward_detector(network *net, unsigned char *CHW, int c, int h, int w, float thresh, float hier_thresh, float nms, float **out, unsigned short *out_len) {
    // we prevent randomness in forward
    srand(0);

    // load and resize image
    image im;
    im.c = c;
    im.h = h;
    im.w = w;
    im.data = calloc(c*h*w, sizeof(float));
    for (int i = 0; i < c*h*w; i++) {
        im.data[i] = CHW[i] / 255.;
    }

    image sized = letterbox_image(im, net->w, net->h);

    // predict
    network_predict(net, sized.data);

    // retrieve detections
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);

    // nms sort
    layer l = net->layers[net->n-1];
    do_nms_sort(dets, nboxes, l.classes, nms);

    // filter results
    float *results = calloc(nboxes * 6, sizeof(float));
    for (int i = 0; i < nboxes; ++i) {
        box bbox = dets[i].bbox;

        // for this detection find the class with maximal probability
        int class = -1;
        float best_prob = 0;
        for (int j = 0; j < l.classes; ++j) {
            float prob = dets[i].prob[j];

            if (prob <= thresh) {
                continue;
            }

            if (class < 0 || best_prob < prob) {
                class = j;
                best_prob = dets[i].prob[j];
            }
        }

        // failed to find a class
        if (class < 0) {
            continue;
        }

        // calculate position for four corners in pixel coordinate.
        float left  = (bbox.x - bbox.w/2.) * im.w;
        float right = (bbox.x + bbox.w/2.) * im.w;
        float top = (bbox.y - bbox.h/2.) * im.h;
        float bottom = (bbox.y + bbox.h/2.) * im.h;

        // set result
        results[*out_len] = class;
        results[*out_len+1] = best_prob;
        results[*out_len+2] = left;
        results[*out_len+3] = top;
        results[*out_len+4] = right;
        results[*out_len+5] = bottom;
        (*out_len) += 6;
    }

    // assign the output
    *out = calloc(*out_len, sizeof(float));
    memcpy(*out, results, *out_len * sizeof(float));

    // clean
    free(results);
    free_detections(dets, nboxes);
    free_image(im);
    free_image(sized);
}

void free_detector(network *net) {
    free_network(net);
    cuda_reset();
}
