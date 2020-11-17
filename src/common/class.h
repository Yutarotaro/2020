#pragma once


#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>


class Camera
{
public:
    Camera();
    virtual ~Camera();

    virtual void init(void);
};
