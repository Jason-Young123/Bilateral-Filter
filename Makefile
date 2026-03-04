WDIR = .
SRC_DIR = $(WDIR)/src
OBJ_DIR = $(WDIR)/obj
INCLUDES_DIR = $(WDIR)/include
ANALYSIS_DIR = $(WDIR)/analysis

#工具链
PREFIX = /usr/local/cuda-12.8/bin/
CXX = $(PREFIX)nvcc
GDB = $(PREFIX)cuda-gdb
NCU = $(PREFIX)ncu
NCU-GUI = $(PREFIX)ncu-ui
NSYS = $(PREFIX)nsys

#编译参数
PLATFORM ?= PLATFORM_NVIDIA
CXXFLAGS += -D$(PLATFORM)
CXXFLAGS += -I$(INCLUDES_DIR) #include path
CXXFLAGS += -Xcompiler -std=c++17
CXXFLAGS += -g -G -use_fast_math
CXXFLAGS += -Xcudafe "--diag_suppress=611" -Wno-deprecated-gpu-targets


#检测opencv是否安装
#OPENCV_EXISTS := $(shell pkg-config --exists opencv4 && echo "yes" || echo "no")
#ifeq ($(OPENCV_EXISTS),yes)
#    CXXFLAGS += -DHAS_CV
#    CXXFLAGS_OPENCV = $(shell pkg-config --cflags opencv4)
#    LD_LIBS = $(shell pkg-config --libs opencv4)
#    $(info [INFO] Built-in OpenCV Detected.)
#else
#    CXXFLAGS_OPENCV =
#    LD_LIBS =
#    $(info [WARN] Built-in OpenCV Not Found... Using Hand-written Version Instead.)
#endif


ifeq ($(CONDA_PREFIX),)
    OPENCV_EXISTS := $(shell pkg-config --exists opencv4 && echo "yes" || echo "no")
    USE_CONDA_PATH := no
else
    OPENCV_EXISTS := $(shell [ -d $(CONDA_PREFIX)/include/opencv4 ] && echo "yes" || echo "no")
    USE_CONDA_PATH := yes
endif


ifeq ($(OPENCV_EXISTS),yes)
    CXXFLAGS += -DHAS_CV
    ifeq ($(USE_CONDA_PATH),yes)
        $(info [INFO] Using OpenCV from Conda Environment: $(CONDA_PREFIX))
        CXXFLAGS_OPENCV = -I$(CONDA_PREFIX)/include/opencv4
        LD_LIBS = -L$(CONDA_PREFIX)/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
		RUNTIME_LIBS := $(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)
    else
        $(info [INFO] Using System Built-in OpenCV via pkg-config.)
        CXXFLAGS_OPENCV = $(shell pkg-config --cflags opencv4)
        LD_LIBS = $(shell pkg-config --libs opencv4)
		RUNTIME_LIBS := $(LD_LIBRARY_PATH)
    endif
else
    CXXFLAGS_OPENCV =
    LD_LIBS =
    $(info [WARN] OpenCV Not Found. Disabling CV-related features.)
endif


SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))
EXEC = $(WDIR)/runTester


#link: .o -> exec
$(EXEC): $(OBJS)
	@echo "[INFO] $^ -> $@"
	@$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPENCV) -o $@ $^ $(LD_LIBS)

#compile: .cu -> .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	@echo "[INFO] $< -> $@"
	@$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPENCV) -c -o $@ $<


run: $(EXEC)
	@LD_LIBRARY_PATH=$(RUNTIME_LIBS) ./$<

gdb: $(EXEC)
	@$(GDB) $<




#Nsight System,系统级分析工具
nsys: $(EXEC)
	@mkdir -p $(ANALYSIS_DIR)
	@$(NSYS) profile -t cuda,nvtx,osrt -o $(ANALYSIS_DIR)/$(EXEC) -f true $<
	@$(NSYS) stats $(ANALYSIS_DIR)/$(EXEC).nsys-rep --force-export=true

#Nsight Compute,内核级分析工具
ncu: $(EXEC)
	@mkdir -p $(ANALYSIS_DIR)
	@$(NCU) --print-details all --nvtx --call-stack --set full $<

ncu-gui: $(EXEC)
	@mkdir -p $(ANALYSIS_DIR)
	@$(NCU) --nvtx --call-stack --set full -f --export $(ANALYSIS_DIR)/$(EXEC).ncu-rep $<
	@$(NCU-GUI) $(ANALYSIS_DIR)/$(EXEC).ncu-rep


clean:
	rm -rf $(OBJ_DIR) $(ANALYSIS_DIR) $(EXEC)



.PHONY: