CXX:= g++ -std=c++17

CFLAGS:= -Wall -O3 -shared -fPIC

CFLAGS+= -I/opt/nvidia/deepstream/deepstream-5.1/sources/includes \
	  	 -I/usr/local/cuda/include

LIBS:= -lnvinfer -lnvparsers

LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_custombboxparser_centerface.cpp
TARGET_LIB:= libnvds_infercustomparser_centerface.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CXX) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
