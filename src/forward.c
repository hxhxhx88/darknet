#include "network.h"

void create_detector(char *cfgfile, char *weightfile, network **net) {
    // load the network
    *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(*net, 1);
}

void forward_detector(network *net, char *filename, float thresh, float hier_thresh, float nms, float **out, unsigned short *out_len) {
    // we prevent randomness in forward
    srand(0);

    // load and resize image
    image im = load_image_color(filename, 0, 0);
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
        results[i * 6] = class;
        results[i * 6+1] = best_prob;
        results[i * 6+2] = left;
        results[i * 6+3] = top;
        results[i * 6+4] = right;
        results[i * 6+5] = bottom;
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
}
