#include <SFML/Graphics.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
//#include <torch/script.h>
#include <iostream>
#include <vector>

int main()
{
    //// Load model
    //torch::jit::script::Module* model = new torch::jit::script::Module;
    //*model = torch::jit::load("/home/rwang7839/kart-sd/cinfer/models/cvs.pt");
    //torch::NoGradGuard no_grad;
    //model->eval();

    // Init display
    sf::RenderWindow window(sf::VideoMode(505, 655), "");
    sf::RectangleShape car(sf::Vector2f(35, 55));
    car.setFillColor(sf::Color::Black);
    car.setPosition(235, 600);

    // Init video captures
    cv::Mat img0;
    cv::VideoCapture cap0;
    cap0.open(0);
    cap0.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap0.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat img1;
    cv::VideoCapture cap1;
    cap1.open(1);
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat img2;
    cv::VideoCapture cap2;
    cap2.open(2);
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat img3;
    cv::VideoCapture cap3;
    cap3.open(3);
    cap3.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap3.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat img4;
    cv::VideoCapture cap4;
    cap4.open(4);
    cap4.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap4.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

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

    //at::Tensor* x0 = new at::Tensor;

    while (window.isOpen())
    {
         cap0.read(img0);
         cv::imshow("0", img0);
         cap1.read(img1);
         cv::imshow("1", img1);
         cap2.read(img2);
         cv::imshow("2", img2);
         cap3.read(img3);
         cv::imshow("3", img3);
         cap4.read(img4);
         cv::imshow("4", img4);

        // cv::Mat crop0 = img0(cv::Range(50, 192), cv::Range(0, 256));
        // cv::Mat crop1 = img1(cv::Range(85, 144), cv::Range(0, 176));
        // cv::Mat crop2 = img2(cv::Range(85, 144), cv::Range(0, 176));

        // std::cout << crop0.rows << crop0.cols << "\n";

        //* x0 = torch::from_blob(img0->data, { 1, 3, img0->rows, img0->cols }, at::kByte);
        // x1 = torch::from_blob(crop1->data, {59, 176, 3}, at::kByte);
        // x2 = torch::from_blob(crop2->data, {59, 176, 3}, at::kByte);

        // std::vector<torch::jit::IValue> *inputs = new std::vector<torch::jit::IValue>;
        // inputs->push_back(x0);
        // inputs->push_back(x1);
        // inputs->push_back(x2);

        // at::Tensor vs_pred = model->forward(*inputs).toTensor();

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // window.clear(sf::Color::White);
        // window.draw(car);
        // for (int row = 0; row < 120; row++){
        //     for (int pos = 0; pos < 101; pos++){
        //         float val = vs_pred.index({0, 101*row + pos}).item<float>();

        //         float x = 255*(1 - val);
        //         if (x < 0){
        //             x = 0;
        //         }
        //         else if (x > 255){
        //             x = 255;
        //         }

        //         sf::RectangleShape rect(sf::Vector2f(5, 5));
        //         sf::Color color(255, x, x);
        //         rect.setFillColor(color);
        //         rect.setPosition(5*pos, 595-(5*row));
        //         window.draw(rect);
        //     }
        // }
        // window.display();

    //    if (cv::wa/*itKey(1) >= 0)
    //        break;
    }

    return 0;
}
