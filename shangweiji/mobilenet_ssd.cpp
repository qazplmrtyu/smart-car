/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2019, Open AI Lab
 * Author: zhangbing@openailab.com
 */

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
void post_process_ssd(cv::Mat img, float threshold, float* outdata, int num)
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
            printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
            printf("BOX:( %g , %g ),( %g , %g )\n", box.x0, box.y0, box.x1, box.y1);
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

int main(int argc, char* argv[])
{
    int ret = -1;
    std::string model_file = "models/mssd.tmfile";

    // init tengine
    if(init_tengine() < 0)
    {
        std::cout << " init tengine failed\n";
        return 1;
    }
    // create graph
    graph_t graph = create_graph(nullptr, "tengine", model_file.c_str());

    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << " ,errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    // input size define
    int img_h = 300;
    int img_w = 300;
    int channel = 3;
    int img_size = img_h * img_w * channel;

    // allocate input_data memory
    float* input_data = ( float* )malloc(sizeof(float) * img_size);

    int node_idx = 0;
    int tensor_idx = 0;

    // get input tensor information
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(input_tensor == nullptr)
    {
        std::printf("Cannot find input tensor,node_idx: %d,tensor_idx: %d\n", node_idx, tensor_idx);
        return -1;
    }

    int dims[] = {1, channel, img_h, img_w};
    // set input tensor size information
    set_tensor_shape(input_tensor, dims, 4);

    // Prerun grapn that allocating each node memory of tengine graph
    ret = prerun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Prerun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    cv::Mat frame;

    // Connecting camera with suitable port
    cv::VideoCapture capture(0);
    
    while(1){
        // Get each frame from camera
	    capture >> frame;
        // Dealing with each frame data
	    get_input_data_ssd(frame, input_data, img_h, img_w);
        // Set input_data into tengine graph
        set_tensor_buffer(input_tensor, input_data, img_size*4);
        // Run graph	
	    run_graph(graph, 1);
        
        // Get output tensor inforamtion
	    tensor_t out_tensor = get_graph_output_tensor(graph, 0, 0);    
        int out_dim[4];
        // Get output tensor size
        ret = get_tensor_shape(out_tensor, out_dim, 4);
        if(ret <= 0)
        {
            std::cout << "get tensor shape failed, errno: " << get_tengine_errno() << "\n";
            return 1;
        }
        // Get output tensor data
        float* outdata = ( float* )get_tensor_buffer(out_tensor);
        int num = out_dim[1];
        float show_threshold = 0.5;
        // Dealing with output data
        post_process_ssd(frame, show_threshold, outdata, num);
        // free output tensor memory
        release_graph_tensor(out_tensor);

        // Show each frame
	    imshow("Mssd", frame);
	    cv::waitKey(1);
    }

    // Release memory for input tensor
    release_graph_tensor(input_tensor);
    // Release memory for each node memory of tengine graph
    ret = postrun_graph(graph);
    if(ret != 0)
    {
        std::cout << "Postrun graph failed, errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    // free memory for input data
    free(input_data);
    // destory tengine graph
    destroy_graph(graph);
    // release all memory of tenging that allocate at beginning 
    release_tengine();

    return 0;
}

