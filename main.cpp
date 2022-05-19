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

const int numTrials = 10;
unsigned long startTime;

void runCycle(double learnRate, int batchSize, double momentum){
    Agent net;
    net.initInput(1, 10, 10);
    net.addConvLayer(4, 10, 10, 3, 3);
    net.addConvLayer(6, 10, 10, 3, 3);
    net.addPoolLayer(6, 5, 5);
    net.addDenseLayer(60);
    net.addDenseLayer(1);
    net.quickSetup();
    
    Grid G(10, 10);
    double sumFinal = 0;
    double avgError = -1;
    for(int k=0; k<numTrials; k++){
        net.randomize(0.2);
        double error = 0;
        for(int t=0; t<=60000; t++){
            for(int i=0; i<batchSize; i++){
                G.randomize(0.5);
                G.inputAgent(&net);
                net.expected = G.byRow();
                net.backProp();
                error += squ(net.output - net.expected);
            }
            if(t>0 && t%1000 == 0){
                avgError = error / batchSize / 1000;
                cout<<t<<' '<<avgError<<" "<<(time(NULL) - startTime)<<'\n';
                if(t%10000 == 0){
                    trialLog(to_string(avgError) + " ");
                }
                error = 0;
            }
            net.updateParameters(learnRate / batchSize, momentum);
        }
        sumFinal += avgError;
        trialLog("TIME: " + to_string(time(NULL) - startTime));
        trialLog("\n");
    }
    trialLog("Final average: " + to_string(sumFinal / numTrials) + "\n\n");
    
    net.save();
    net.close();
}

int main(int argc, const char * argv[]) {
    srand((unsigned) time(NULL));
    startTime = time(NULL);
    /*
    testDeterministic();
     */
    double learnRate = 6e-06;
    int batchSize = 30;
    double momentum = 0.9;
    runCycle(learnRate, batchSize, momentum);
    trialLog("Changing learnRate to:\n");
    for(auto learnRate : {1e-05, 6e-06, 4e-06, 2e-06}){
        trialLog(to_string(learnRate) + ":\n");
        runCycle(learnRate, batchSize, momentum);
    }
    trialLog("Changing batchSize to:\n");
    for(auto batchSize : {1, 10, 30, 60}){
        trialLog(to_string(batchSize) + ":\n");
        runCycle(learnRate, batchSize, momentum);
    }
    trialLog("Changing momentum to:\n");
    for(auto momentum : {0., 0.2, 0.5, 0.9, 0.95}){
        trialLog(to_string(momentum) + ":\n");
        runCycle(learnRate, batchSize, momentum);
    }
}
