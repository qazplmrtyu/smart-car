#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "anglecontrol.h"//舵机pid控制
#include "ssd.h"//目标检测
#include <vector>
#include "tengine_c_api.h"//执行tenggine的接口

#include <string>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <errno.h>
#include <unistd.h> 
#include <wiringPi.h>
#include <wiringSerial.h> //树莓派串口

#define UART_DEVICE "/dev/ttyS1"
#define HIGHH 640 //图像大小
#define WIDTH 640
extern float KP, KI, KD;

using namespace std;
using namespace cv;

//绝对值
int jueduizhi(int error){
	if (error < 0) error = -error;
	return error;

}

//寻找图像中间一行白色的中点
int chazhao(int mid[],int midpoint_last){
	int zuo = 0,you = WIDTH;
	for(int i = midpoint_last; i >= 20; i--){
			if (mid[i] == 0){
				int sumzuo = 0;
				for(int j = i; j >= i-15; j--) sumzuo += mid[j];
				if (sumzuo <= 1000) {zuo = i; break;}
			}
		}
		for(int i = midpoint_last; i <= WIDTH - 20; i++){
			if (mid[i] == 0){
				int sumyou = 0;
				for(int j = i; j <= i + 15; j++) sumyou += mid[j];
				if (sumyou <= 1000) {you = i; break;}
			}
		}
		int midpoint = (zuo + you) / 2;
		
	return midpoint;

}

//膨胀

Mat peformErosion(Mat inputImage, int erosionElement, int erosionSize){
	Mat outputImage;
	int erosionType;
	
	if(erosionElement == 0) erosionType = MORPH_RECT;
	if(erosionElement == 1) erosionType = MORPH_CROSS;
	if(erosionElement == 1) erosionType = MORPH_ELLIPSE;

	Mat element = getStructuringElement(erosionType, Size(2*erosionSize + 1, 2* erosionSize +1),
			Point(erosionSize, erosionSize));
	erode(inputImage, outputImage, element);
	
	return outputImage;
}

//腐蚀
Mat peformDilation(Mat inputImage, int dilationElement, int dilationSize){
	Mat outputImage;
	int dilationType;
	
	if(dilationElement == 0) dilationType = MORPH_RECT;
	if(dilationElement == 1) dilationType = MORPH_CROSS;
	if(dilationElement == 1) dilationType = MORPH_ELLIPSE;

	Mat element = getStructuringElement(dilationType, Size(2*dilationSize + 1, 2* dilationSize +1),
			Point(dilationSize, dilationSize));
	dilate(inputImage, outputImage, element);
	
	return outputImage;
}

int main(void)
{	int sudu = 20;
	// Kalman
	const int stateNum = 4;
	const int measureNum = 2;                                    	
	KalmanFilter KF(stateNum, measureNum, 0);//卡尔曼滤波器
	Mat XXX = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1 );
	KF.transitionMatrix = XXX;  
	setIdentity(KF.measurementMatrix);                                           
	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));                           
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));                        
	setIdentity(KF.errorCovPost, Scalar::all(1));                                  
	randn(KF.statePost, Scalar::all(0), Scalar::all(0.1)); 
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
	// Kalman
        int mode = 0;
        KP = 0.4;
        KI = 0;
        cin >> mode;
        //模式一二值化寻迹
        if (mode == 1)
        {
    	VideoCapture cap(0);
    	if(!cap.isOpened())
   	 {
        	cout << "Cannot open a camera" << endl;
        
    	}
	//串口初始化
	int fd;
	if ((fd = serialOpen(UART_DEVICE, 9600)) < 0)
	{
		fprintf(stderr, "Unable to open serial device: %s\n", strerror(errno));
		return 1 ;
	}
	
 	int midpoint = WIDTH / 2;	

    	/* Frame capture and image processing */
    	for(;;)
    	{
        	Mat frame, dst;
        	cap >> frame;
        	if (frame.empty())
        	{
            	cout << "End of video stream" << endl;
            		break;
        	}

        	cvtColor(frame, dst, COLOR_BGR2GRAY);//灰度转化
        	threshold(dst, dst, 125, 255, THRESH_BINARY);//阈值分割
            //膨胀腐蚀
		dst = peformErosion(dst, 1, 5);
		dst = peformDilation(dst, 1, 5);
		resize(dst, dst, Size(WIDTH, HIGHH));

		int mid[WIDTH];
		for(int i = 0; i <= dst.rows -1; i++){
			mid[i] = (int) dst.at<uchar>(HIGHH/2, i);
		}
		
		
        //卡尔曼滤波修正中点
		midpoint = chazhao(mid, midpoint);
                //2.kalman prediction
                Mat prediction = KF.predict();

                //3.update measurement
                measurement.at<float>(0) = (float)midpoint;
                measurement.at<float>(1) = (float)WIDTH;

                //4.update
                KF.correct(measurement);
                midpoint = (int)measurement.at<float>(0);
                int error = WIDTH / 2 - (int)measurement.at<float>(0);
                error = jueduizhi(error);
                int steer = duojictrl(error, (int)measurement.at<float>(0));


                //串口报文定义
                int FLAG = 6;

                char info[8], info2[3], info3[2];
                sprintf(info, "%d", FLAG);
                sprintf(info2, "%03d", sudu);
                sprintf(info3, "%02d", steer);
                strcat(info, info2);
                strcat(info, info3);
                info[6] = '#'; info[7] = '\0';

                //串口输出
                serialPuts(fd, info);
                cout << info << endl;
                Point p(measurement.at<float>(0), HIGHH/2);

                circle(dst, p, 5, Scalar(0), 2, CV_8U);

                imshow("Processed Frames", dst);

                /* Exit when Esc is detected */
                char c = (char)waitKey(30);
                if (c == 27)
                { serialClose(fd); break;}

        }
        }

        //模式二图像识别跟随
        if (mode == 2)
        {
            int midpoint = WIDTH / 2;

            int ret = -1;
            std::string model_file = "./ssd.tmfile";

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
                int point;
                point = post_process_ssd(frame, show_threshold, outdata, num);
                midpoint = point;

                    //2.kalman prediction
                    Mat prediction = KF.predict();

                    //3.update measurement
                    measurement.at<float>(0) = (float)midpoint;
                    measurement.at<float>(1) = (float)WIDTH;

                    //4.update
                    KF.correct(measurement);
                    midpoint = (int)measurement.at<float>(0);
                    int error = WIDTH / 2 - (int)measurement.at<float>(0);
                    error = jueduizhi(error);
                    int steer = duojictrl(error, (int)measurement.at<float>(0));
                    int fd;
                    if ((fd = serialOpen(UART_DEVICE, 9600)) < 0)
                    {
                            fprintf(stderr, "Unable to open serial device: %s\n", strerror(errno));
                            return 1 ;
                    }


                    int FLAG = 6;

                    char info[8], info2[3], info3[2];
                    sprintf(info, "%d", FLAG);
                    sprintf(info2, "%03d", sudu);
                    sprintf(info3, "%02d", steer);
                    strcat(info, info2);
                    strcat(info, info3);
                    info[6] = '#'; info[7] = '\0';

                    serialPuts(fd, info);
                    cout << info << endl;


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

            // free memory for input data
            free(input_data);
            // destory tengine graph
            destroy_graph(graph);
            // release all memory of tenging that allocate at beginning
            release_tengine();


        }
	return 0;
}
