
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include "tengine_c_api.h"

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

/*
*   Pre processing for mobilenet_ssd detection input 
*/

void get_input_data_ssd(cv::Mat img, float* input_data, int img_h, int img_w)
{
    // Resize the size so that it can suitable for model's input size
    cv::resize(img, img, cv::Size(img_h, img_w));
    // Convert image data from uint8 to float
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;
    // Pre processing for mobilenet_ssd input data
    float mean[3] = {127.5, 127.5, 127.5};
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843 * (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

/*
*   Post processing for mobilenet_ssd detection
*   including draw box and label targets
*/
int post_process_ssd(cv::Mat img, float threshold, float* outdata, int num)
{
     const char* class_names[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                 "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                 "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                 "sofa",       "train",     "tvmonitor"};

    // const char* class_names[] = {"background",  "person"};

    int raw_h = img.size().height;
    int raw_w = img.size().width;
    std::vector<Box> boxes;
    int line_width = raw_w * 0.005;
    for(int i = 0; i < num; i++)
    {
        // Compared with threshold, keeping the result if the value larger than threshold, otherwise
        // delete the result
        if(outdata[1] >= threshold)
        {
            Box box;
            box.class_idx = outdata[0];
            box.score = outdata[1];
            box.x0 = outdata[2] * raw_w;
            box.y0 = outdata[3] * raw_h;
            box.x1 = outdata[4] * raw_w;
            box.y1 = outdata[5] * raw_h;
            boxes.push_back(box);
            //printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
            //printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
        }
        outdata += 6;
    }

    // drawing boxes and label the target into image
    for(int i = 0; i < ( int )boxes.size(); i++)
    {
        if(boxes[i].class_idx == 15)
        {
            Box box = boxes[i];
            cv::rectangle(img, cv::Rect(box.x0, box.y0, (box.x1 - box.x0), (box.y1 - box.y0)), cv::Scalar(255, 255, 0),
                          line_width);
            return int(( box.x0 + box.x1) / 2 );
            std::ostringstream score_str;
            score_str << box.score;
            std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(img, cv::Rect(cv::Point(box.x0, box.y0 - label_size.height),
                               cv::Size(label_size.width, label_size.height + baseLine)),
                               cv::Scalar(255, 255, 0), CV_FILLED);
            cv::putText(img, label, cv::Point(box.x0, box.y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
}

