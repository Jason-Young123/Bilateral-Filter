#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>



/*
读取image图像,并将相关像素信息写入filename
例如: image = "resource/lena.png", filename = "resource/lena.hex"
写入filename的信息依次包括: 图片宽, 高, 通道数(1或3,黑白/RGB),按行主序存储的每个像素点信息, RGB或灰度
*/
bool dataGen(const char* imgPath, const char* outBinPath){
    //step1: 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    if(img.empty()){
        std::cerr << "Error: Could not open image: " << imgPath << std::endl;
        return false;
    }    

    //step2: 获取图像基本属性
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    //step3: 打开待写入文件
    std::ofstream outFile(outBinPath, std::ios::out | std::ios::binary);
    if(!outFile.is_open()){
        std::cerr << "Error: Could not open file: " << outBinPath << std::endl;
        return false;
    }

    //step4: 写入数据
    outFile.write(reinterpret_cast<const char*>(&width), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&height), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&channels), sizeof(int));

    cv::Mat finalImg;
    if(channels == 3){
        cv::cvtColor(img, finalImg, cv::COLOR_BGR2RGB);
    }
    else{
        finalImg = img;
    }
    if(!finalImg.isContinuous()){//确保连续存储
        finalImg = finalImg.clone();
    }

    size_t pixelSize = width * height * channels * sizeof(unsigned char);
    outFile.write(reinterpret_cast<const char*>(finalImg.data), pixelSize);

    outFile.close();
    std::cout << "Successfully generated file: " << outBinPath << std::endl;
    return true;
}




int main(){
    dataGen("resource/lena.png", "resource/lena.bin");
    return 0;
}