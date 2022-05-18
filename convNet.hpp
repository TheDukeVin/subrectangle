//
//  convNet.hpp
//  convNet
//
//  Created by Kevin Du on 5/16/22.
//

#ifndef convNet_hpp
#define convNet_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <math.h>
using namespace std;

//training deatils

#define momentum 0.9
#define maxNorm 1
#define batchSize 30

//network details

#define numLayers 5
#define maxNodes 600
#define maxDepth 6
#define maxConvSize 3

double squ(double x);

// For the network
double randWeight(double startingParameterRange);

double nonlinear(double x);

// If f = nonlinear, then this function is (f' \circ f^{-1}).
double dinvnonlinear(double x);

class Layer{
public:
    ifstream* netIn;
    ofstream* netOut;
    int numParams, numWeights, numBias;
    double* params;
    double* weights;
    double* bias;
    double* Dparams;
    double* Dweights;
    double* Dbias;
    virtual void pass(double* inputs, double* outputs){};
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs){};
    virtual void accumulateGradient(double* inputs, double* Doutputs){};
    void setupParams();
    void randomize(double startingParameterRange);
    void resetGradient();
    void updateParameters(double learnRate);
    void save();
    void readNet();
};

class ConvLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int convHeight, convWidth;
    int shiftr, shiftc;
    int w1, w2, w3;
    
    ConvLayer(int inD, int inH, int inW, int outD, int outH, int outW, int convH, int convW);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
};

class PoolLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int maxIndices[maxNodes];
    
    PoolLayer(int inD, int inH, int inW, int outD, int outH, int outW);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
};

class DenseLayer : public Layer{
public:
    int inputSize, outputSize;
    
    DenseLayer(int inSize, int outSize);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
};

class Agent{
public:
    Layer* layers[numLayers];
    double activation[numLayers+1][maxNodes];
    double Dbias[numLayers][maxNodes];
    double output;
    double expected;
    
    // For network initiation
    int layerIndex;
    int prevDepth, prevHeight, prevWidth;
    
    // For file I/O
    ifstream* netIn;
    ofstream* netOut;
    
    void setupIO();
    void initInput(int depth, int height, int width);
    void addConvLayer(int depth, int height, int width, int convHeight, int convWidth);
    void addPoolLayer(int depth, int height, int width);
    void addDenseLayer(int numNodes);
    void randomize(double startingParameterRange);
    
    // For network usage and training
    void quickSetup();
    void close();
    void pass();
    void resetGradient();
    void backProp();
    void updateParameters(double learnRate);
    void save();
    void readNet();
};

#endif /* convNet_hpp */