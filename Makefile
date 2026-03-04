WDIR = .
SRC_DIR = $(WDIR)/src
OBJ_DIR = $(WDIR)/obj
INCLUDES_DIR = $(WDIR)/include
ANALYSIS_DIR = $(WDIR)/analysis
ENV_DIR = $(WDIR)/tmp_env



PLATFORM        ?= nvidia
PLATFORM_DEFINE ?= -DPLATFORM_NVIDIA
STUDENT_SUFFIX  := cu
CFLAGS          := -std=c++17 -O0
EXTRA_LIBS     	:= 

# Compiler & Tester object selection based on PLATFORM
ifeq ($(PLATFORM),nvidia)
    PLATFORM_DEFINE 	:= -DPLATFORM_NVIDIA
    PREFIX 		:= /usr/local/cuda-12.8/bin/
    CXX 		:= $(PREFIX)nvcc
    GDB 		:= $(PREFIX)cuda-gdb
    NCU 		:= $(PREFIX)ncu
    NCU-GUI 		:= $(PREFIX)ncu-ui
    NSYS 		:= $(PREFIX)nsys
    CXXFLAGS		:= -Xcompiler -std=c++17 -g -G -use_fast_math -Xcudafe "--diag_suppress=611" -Wno-deprecated-gpu-targets -I$(INCLUDES_DIR)
    SUFFIX		:= cu
else ifeq ($(PLATFORM),iluvatar)
    CC          	:= clang++
    CFLAGS          := -std=c++17 -O3
    TEST_OBJ    	:= tester/tester_iluvatar.o
    PLATFORM_DEFINE := -DPLATFORM_ILUVATAR
    EXTRA_LIBS		:= -lcudart -I/usr/local/corex/include -L/usr/local/corex/lib64 -fPIC
else ifeq ($(PLATFORM),moore)
    PLATFORM_DEFINE 	:= -DPLATFORM_MOORE
    CXX          	:= mcc
    CXXFLAGS          	:= -std=c++17 -O3 -I$(INCLUDES_DIR) -I/usr/local/musa/include 
    LIBS		:= -L/usr/lib/gcc/x86_64-linux-gnu/11/ -L/usr/local/musa/lib -lmusart
    RUNTIME_LIBS	:= /usr/local/musa/lib
    SUFFIX  		:= mu
else ifeq ($(PLATFORM),metax)
    CC          	:= mxcc
    TEST_OBJ    	:= tester/tester_metax.o
	STUDENT_SUFFIX  := maca
	PLATFORM_DEFINE := -DPLATFORM_METAX
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, moore, metax))
endif




#工具链

#编译参数
#PLATFORM ?= PLATFORM_NVIDIA
#CXXFLAGS += -D$(PLATFORM)
#CXXFLAGS += -I$(INCLUDES_DIR) #include path
#CXXFLAGS += -Xcompiler -std=c++17
#CXXFLAGS += -g -G -use_fast_math
#CXXFLAGS += -Xcudafe "--diag_suppress=611" -Wno-deprecated-gpu-targets


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


SRCS = $(wildcard $(SRC_DIR)/*.$(SUFFIX))
OBJS = $(patsubst $(SRC_DIR)/%.$(SUFFIX), $(OBJ_DIR)/%.o, $(SRCS))
EXEC = $(WDIR)/runTester


#(当前路径下如果有tmp_env路径则全部删除)用conda在当前路径下创建虚拟环境tmp_env, 下载安装opencv, 然后将CXXFLAGS_OPENCV和LD_LIBS指向相应路径
#完成功能: 检测是否有内嵌opencv: $(shell pkg-config --exists opencv4), 如果有, 则echo 相应info, CXXFLAGS += -DHAS_CV, CXXFLAGS_OPENCV = $(shell pkg-config --cflags opencv4)
#        LD_LIBS = $(shell pkg-config --libs opencv4)
#	else, 如果有conda: 在$(ENV_DIR)下创建虚拟环境,并安装cpp版的opencv, echo相应info, CXXFLAGS += -DHAS_CV, 相应地设置CXXFLAGS_OPENCV和LD_LIBS
#	else, echo相应info, CXXFLAGS不动, CXXFLAGS_OPENCV和LD_LIBS设置为空
init:
	
	@rm -rf $(ENV_DIR)
	@echo "[INFO] Create virtual environment for opencv."
	




#link: .o -> exec
$(EXEC): $(OBJS)
	@echo "[INFO] $^ -> $@"
	@$(CXX) $(CXXFLAGS) $(LIBS) $(CXXFLAGS_OPENCV) -o $@ $^ $(LD_LIBS)

#compile: .cu -> .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.$(SUFFIX)
	@mkdir -p $(OBJ_DIR)
	@echo "[INFO] $< -> $@"
	@$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPENCV) $(PLATFORM_DEFINE) -c -o $@ $<


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
