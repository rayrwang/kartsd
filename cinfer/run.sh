g++ -c main.cpp
g++ main.o -o main-app -lsfml-graphics -lsfml-window -lsfml-system -lopencv_core -lopencv_videoio -lopencv_highgui -lbackend_with_compiler -lc10 -ltorch -ltorch_cpu -ltorch_global_deps
./main-app
