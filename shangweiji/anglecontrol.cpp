#include "anglecontrol.h"
#include <iostream>



#define STEER_MIN 9
#define STEER_MAX 81
#define STEER_MID 45

using namespace std;

float KP = 0.52, KI = 0, KD = 0;


int pid_ctrl(int error){
	static float Bias, Pwm, Intergral_bias, Last_bias;
	Bias = error;
	Intergral_bias += Bias;
	Pwm = KP * Bias + KI * Intergral_bias + KD * (Bias - Last_bias);
	Last_bias = Bias;

	return Pwm;
}
int duojictrl(int error, int midpoint){
	int pid = pid_ctrl(error);
	int steer;
	if (midpoint > 320) steer = STEER_MID - pid;
	else steer = STEER_MID + pid;

	if (steer > STEER_MAX) steer = STEER_MAX;
	if (steer < STEER_MIN) steer = STEER_MIN;

	return steer;  

}