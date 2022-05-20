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
#include <string>
using namespace std;

//training deatils

// learnRate, momentum, and batchSize are changed to function arguments
// instead of constants.
#define maxNorm 1

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
    void updateParameters(double mult, double momentum);
    void save();
    void readNet();
    
    virtual ~Layer(){}
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
    
    virtual ~ConvLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class PoolLayer : public Layer{
public:
    int inputDepth, inputHeight, inputWidth;
    int outputDepth, outputHeight, outputWidth;
    int* maxIndices;
    
    PoolLayer(int inD, int inH, int inW, int outD, int outH, int outW);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    
    virtual ~PoolLayer(){
        //delete[] params;
        //delete[] Dparams;
        delete[] maxIndices;
    }
};

class DenseLayer : public Layer{
public:
    int inputSize, outputSize;
    
    DenseLayer(int inSize, int outSize);
    
    virtual void pass(double* inputs, double* outputs);
    virtual void backProp(double* inputs, double* Dinputs, double* Doutputs);
    virtual void accumulateGradient(double* inputs, double* Doutputs);
    
    virtual ~DenseLayer(){
        delete[] params;
        delete[] Dparams;
    }
};

class Agent{
public:
    unsigned long numLayers;
    unsigned maxNodes = 0;
    Layer** layers; // keep an array of pointers, since abstract classes need to be accessed by reference.
    double** activation;
    double** Dbias;
    double output;
    double expected;
    
    // For file I/O
    ifstream netIn;
    ofstream netOut;
    
    void initInput(int depth, int height, int width);
    void addConvLayer(int depth, int height, int width, int convHeight, int convWidth);
    void addPoolLayer(int depth, int height, int width);
    void addDenseLayer(int numNodes);
    void randomize(double startingParameterRange);
    
    // For network usage and training
    void quickSetup();
    void pass();
    void resetGradient();
    void backProp();
    void updateParameters(double mult, double momentum);
    void save();
    void readNet();
    ~Agent(){
        for(int i=0; i<numLayers; i++){
            delete layers[i];
            delete[] Dbias[i];
        }
        for(int i=0; i<=numLayers; i++){
            delete[] activation[i];
        }
    }
    
private:
    // For network initiation
    int prevDepth, prevHeight, prevWidth;
    vector<Layer*> layerHold;
    void setupIO();
};

#endif /* convNet_hpp */
