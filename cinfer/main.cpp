#include <SFML/Graphics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <algorithm>

int main()
{
    // Load model
    torch::jit::script::Module *model = new torch::jit::script::Module;
    *model = torch::jit::load("C:/Users/rwang/Documents/kartsd/cinfer/cmodels/cvs.pt");
    torch::NoGradGuard no_grad;
    model->eval();

    // Init display
    sf::RenderWindow window(sf::VideoMode(505, 600), "");
    sf::RectangleShape car(sf::Vector2f(0.9/0.25*5, 1.5/0.25*5));
    car.setFillColor(sf::Color::Black);
    car.setOrigin(car.getSize().x, car.getSize().y);
    car.setPosition(252.5, 400 + (1.5/2 - 0.2)/0.25*5);
    sf::RectangleShape rect(sf::Vector2f(5, 5));
    
    // Init video captures
    cv::Mat *img0 = new cv::Mat;
    cv::VideoCapture cap0;
    cap0.open(0);
    cap0.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap0.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap0.set(cv::CAP_PROP_BUFFERSIZE, 3);
    cap0.set(cv::CAP_PROP_FPS, 1);

    cv::Mat *img1 = new cv::Mat;
    cv::VideoCapture cap1;
    cap1.open(1);
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap1.set(cv::CAP_PROP_BUFFERSIZE, 3);
    cap1.set(cv::CAP_PROP_FPS, 1);

    cv::Mat *img2 = new cv::Mat;
    cv::VideoCapture cap2;
    cap2.open(2);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap2.set(cv::CAP_PROP_BUFFERSIZE, 3);
    cap2.set(cv::CAP_PROP_FPS, 1);

    cv::Mat *img3 = new cv::Mat;
    cv::VideoCapture cap3;
    cap3.open(3);
    cap3.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap3.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap3.set(cv::CAP_PROP_BUFFERSIZE, 3);
    cap3.set(cv::CAP_PROP_FPS, 1);

    cv::Mat *img4 = new cv::Mat;
    cv::VideoCapture cap4;
    cap4.open(4);
    cap4.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap4.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap4.set(cv::CAP_PROP_BUFFERSIZE, 3);
    cap4.set(cv::CAP_PROP_FPS, 1);

    // Init cam windows
    cv::namedWindow("0", cv::WINDOW_NORMAL);
    cv::resizeWindow("0", 240, 180);
    cv::namedWindow("1", cv::WINDOW_NORMAL);
    cv::resizeWindow("1", 240, 180);
    cv::namedWindow("2", cv::WINDOW_NORMAL);
    cv::resizeWindow("2", 240, 180);
    cv::namedWindow("3", cv::WINDOW_NORMAL);
    cv::resizeWindow("3", 240, 180);
    cv::namedWindow("4", cv::WINDOW_NORMAL);
    cv::resizeWindow("4", 240, 180);

    at::Tensor x0;
    at::Tensor x1;
    at::Tensor x2;
    at::Tensor x3;
    at::Tensor x4;

    at::Tensor vsPred;

    while (window.isOpen())
    {
        cap0.grab();
        cap1.grab();
        cap2.grab();
        cap3.grab();
        cap4.grab();
        cap0.retrieve(*img0);
        cap1.retrieve(*img1);
        cap2.retrieve(*img2);
        cap3.retrieve(*img3);
        cap4.retrieve(*img4);
        cv::imshow("0", *img0);
        cv::imshow("1", *img1);
        cv::imshow("2", *img2);
        cv::imshow("3", *img3);
        cv::imshow("4", *img4);

        x0 = torch::from_blob(img0->data, { 1, 3, 480, 640 }, at::kByte);
        x1 = torch::from_blob(img1->data, { 1, 3, 480, 640 }, at::kByte);
        x2 = torch::from_blob(img2->data, { 1, 3, 480, 640 }, at::kByte);
        x3 = torch::from_blob(img3->data, { 1, 3, 480, 640 }, at::kByte);
        x4 = torch::from_blob(img4->data, { 1, 3, 480, 640 }, at::kByte);
        x0 = x0.toType(c10::kFloat);
        x1 = x1.toType(c10::kFloat);
        x2 = x2.toType(c10::kFloat);
        x3 = x3.toType(c10::kFloat);
        x4 = x4.toType(c10::kFloat);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x0);
        inputs.push_back(x1);
        inputs.push_back(x2);
        inputs.push_back(x3);
        inputs.push_back(x4);

        vsPred = model->forward(inputs).toTensor();

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color(210, 210, 210));
        for (int row = 0; row < 120; row++){
            for (int pos = 0; pos < 101; pos++){
                float d = vsPred.index({0, 101*row + pos}).item<float>();
                float e = vsPred.index({ 0, 12120 + 101 * row + pos }).item<float>();

                if (d > 0.4) {
                    d = 210 + 45 * d;
                    d = std::max((float)0, d);
                    d = std::min((float)255, d);
                    rect.setFillColor(sf::Color(d, d, d));
                    rect.setPosition(5 * pos, 595 - (5 * row));
                    window.draw(rect);
                }
                if (e > 0.2) {
                    e = 255*(1 - e);
                    e = std::max((float)0, e);
                    e = std::min((float)255, e);
                    rect.setFillColor(sf::Color(255, e, e));
                    rect.setPosition(5 * pos, 595 - (5 * row));
                    window.draw(rect);
                }
            }
        }
        window.draw(car);
        window.display();
    }

    return 0;
}
