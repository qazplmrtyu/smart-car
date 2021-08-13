
CFLAGS = -g -Wall
LIBS = $(shell pkg-config opencv --cflags --libs)
OBJS = ssd.cpp anglecontrol.cpp main.cpp
CXX = g++
all: control 
control: $(OBJS)
	$(CXX) $(CFLAGS) $(OBJS) $(LIBS) -lwiringPi -lpthread -I/usr/local/AID/Tengine/include/ -L /usr/local/AID/Tengine/lib -l tengine -o $@   


