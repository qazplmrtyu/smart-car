#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <wiringPi.h>
#include <wiringSerial.h>


#define UART_DEVICE "/dev/ttyS1"

int usart(char  *s)
{
		int fd;
		if ((fd = serialOpen(UART_DEVICE, 115200)) < 0)
		{
			fprintf(stderr, "Unable to open serial device: %s\n", strerror(errno));
			return 1 ;
		}
		
		serialPuts(fd, s);

		/*
		for (;;)
		{
			if (serialDataAvail(fd) > 0)
			{
				putchar(serialGetchar(fd));
				fflush (stdout) ;
			}

		}
		return 0;
		*/
}


