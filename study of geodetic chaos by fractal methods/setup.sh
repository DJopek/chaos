#!/bin/zsh

pip install -r requirements.txt

mkdir ./data
mkdir ./build

cd Gravitacek-2

mkdir external
cd external
git clone https://github.com/google/googletest.git -b v1.15.2
cd ..

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native"
cmake --build .

cp -r ./* ../../build/
