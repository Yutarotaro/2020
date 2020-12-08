#include "class.h"
#include <iostream>

Camera::Camera()
{
}

Camera::~Camera()
{
    std::cout << "See you!" << std::endl;
}


void Camera::init(void)
{

    std::cout << "Initialized!" << std::endl;
}

void Camera::pose_estimation(void)
{
}
