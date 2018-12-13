#ifndef FORWARD_H
#define FORWARD_H

// Create a detector handle. `xpu` is the index of the GPU, or -1 indicating CPU.
int create_detector(int xpu, char *cfgfile, char *weightfile, void **handle);

// Run a detection forward. If N objects are detected, `out_len` will be equal to 6N, and the `out` array will consist of N [label, prob, left, top, right, bottom]s.
// Note that the input image must in CHW order.
void forward_detector(void *handle, unsigned char *CHW, int c, int h, int w, float thresh, float hier_thresh, float nms, float **out, unsigned short *out_len);

// Free the detector handle.
void free_detector(void *handle);

#endif

