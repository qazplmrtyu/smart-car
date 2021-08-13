#ifndef _SSD_H_
#define _SSD_H_

void get_input_data_ssd(cv::Mat img, float* input_data, int img_h, int img_w);
int post_process_ssd(cv::Mat img, float threshold, float* outdata, int num);

#endif