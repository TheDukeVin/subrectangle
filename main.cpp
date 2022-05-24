//
//  main.cpp
//  subrectangle
//
//  Created by Kevin Du on 5/15/22.
//

#include "subrect.h"

void testDeterministic(){
    Grid G(3, 4, vector<bool>{
        1, 0, 1, 1,
        1, 1, 1, 0,
        0, 1, 1, 1
    });
    cout<<G.checkAll()<<'\n';
    cout<<G.DP()<<'\n';
    cout<<G.byRow()<<'\n';
    cout<<"Verifying DP\n";
    G.verifyAllDP(50, 0.5, 10);
    cout<<"Verifying byrow\n";
    G.verifyDPRow(50, 0.5, 10);
    int nums[11] = {5, 8, 2, 7, 10, 1, 4, 2, 8, 5, 9};
    MinTable M(11, nums);
    M.verify(100, 100, 10);
}

void trialLog(string s){
    ofstream logOut("log.out", ios::app);
    logOut<<s<<' ';
    logOut.close();
}

unsigned long startTime;

void runCycle(double learnRate, int batchSize, double momentum){
    Agent net;
    net.initInput(1, 10, 10);
    net.addConvLayer(7, 10, 10, 5, 5);
    net.addConvLayer(7, 10, 10, 5, 5);
    net.addPoolLayer(7, 5, 5);
    net.addDenseLayer(120);
    net.addDenseLayer(1);
    net.quickSetup();
    
    net.netIn = ifstream("net.in");
    net.readNet();
    
    Grid G(10, 10);
    double avgError;
    double error = 0;
    
    list<double> errorQueue;
    int queueSize = 50;
    for(int i=0; i<queueSize; i++){
        errorQueue.push_back(10000);
    }
    double minError = 10000;
    
    for(int t=0; t<=1200000; t++){
        double batchCost = 0;
        for(int i=0; i<batchSize; i++){
            G.randomize(0.5);
            G.inputAgent(&net);
            net.expected = G.byRow();
            net.backProp();
            double cost = squ(net.output - net.expected);
            error += cost;
            batchCost += cost;
        }
        if(t>0 && t%1000 == 0){
            avgError = error / batchSize / 1000;
            cout<<t<<' '<<avgError<<" "<<(time(NULL) - startTime)<<'\n';
            if(t%10000 == 0){
                trialLog(to_string(avgError) + " ");
            }
            error = 0;
        }
        
        errorQueue.pop_front();
        errorQueue.push_back(batchCost/batchSize);
        double currError = 0;
        for(auto it=errorQueue.begin(); it!=errorQueue.end(); it++){
            currError += *it;
        }
        currError /= queueSize;
        if(currError < minError){
            minError = currError;
            cout<<"Saved to "<<minError<<" "<<(time(NULL) - startTime)<<'\n';
            net.netOut = ofstream("net.out");
            net.save();
            net.netOut.close();
        }
        
        net.updateParameters(learnRate / batchSize, momentum);
    }
    trialLog("TIME: " + to_string(time(NULL) - startTime));
    trialLog("\n");
    
    //net.save();
}

void evaluate(){
    Agent net;
    net.initInput(1, 10, 10);
    net.addConvLayer(7, 10, 10, 5, 5);
    net.addConvLayer(7, 10, 10, 5, 5);
    net.addPoolLayer(7, 5, 5);
    net.addDenseLayer(120);
    net.addDenseLayer(1);
    net.quickSetup();
    net.netIn = ifstream("net.in");
    net.readNet();
    
    Grid G(10, 10);
    ofstream fout("examples.out");
    for(int t=0; t<10; t++){
        G.randomize(0.5);
        for(int i=0; i<10; i++){
            for(int j=0; j<10; j++){
                if(G.A[i][j]) fout<<"*";
                else fout<<'.';
            }
            fout<<'\n';
        }
        fout<<G.byRow()<<'\n';
        fout<<G.evalAgent(&net)<<"\n\n";
        G.inputAgent(&net);
        net.pass();
        fout<<net.output<<"\n\n";
    }
    double error = 0;
    double numCorrect = 0;
    int trials = 100000;
    for(int i=0; i<trials; i++){
        G.randomize(0.5);
        double expected = G.byRow();
        //double agentOutput = G.evalAgent(&net);
        G.inputAgent(&net);
        net.pass();
        double agentOutput = net.output;
        error += squ(expected - agentOutput);
        numCorrect += abs(expected - agentOutput) < 0.5;
    }
    cout<<"Average error: "<<(error / trials)<<'\n';
    cout<<"Accuracy: "<<(numCorrect / trials)<<'\n';
}

int main(int argc, const char * argv[]) {
    srand((unsigned) time(NULL));
    startTime = time(NULL);
    /*
    testDeterministic();
     */
    /*
    double learnRate = 6e-06;
    int batchSize = 30;
    double momentum = 0.9;
    runCycle(learnRate, batchSize, momentum);
    */
    
    evaluate();
    
}
