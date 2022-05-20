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
    net.addConvLayer(4, 10, 10, 5, 5);
    net.addConvLayer(6, 10, 10, 5, 5);
    net.addPoolLayer(6, 5, 5);
    net.addDenseLayer(60);
    net.addDenseLayer(1);
    net.quickSetup();
    
    Grid G(10, 10);
    double sumFinal = 0;
    double avgError = -1;
    for(int k=0; k<1; k++){
        net.randomize(0.2);
        double error = 0;
        for(int t=0; t<=600000; t++){
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
}

int main(int argc, const char * argv[]) {
    srand((unsigned) time(NULL));
    startTime = time(NULL);
    /*
    testDeterministic();
     */
    /*
    double learnRate = 1e-05;
    int batchSize = 30;
    double momentum = 0.9;
    runCycle(learnRate, batchSize, momentum);
    */
    Agent net;
    net.initInput(1, 10, 10);
    net.addConvLayer(4, 10, 10, 5, 5);
    net.addConvLayer(6, 10, 10, 5, 5);
    net.addPoolLayer(6, 5, 5);
    net.addDenseLayer(60);
    net.addDenseLayer(1);
    net.quickSetup();
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
        /*
        G.inputAgent(&net);
        net.pass();
        fout<<net.output<<"\n\n";*/
    }
    /*
    double error = 0;
    int trials = 1000;
    for(int i=0; i<trials; i++){
        G.randomize(0.5);
        error += squ(G.byRow() - G.evalAgent(&net));
    }
    cout<<"Average error: "<<(error / trials)<<'\n';*/
}
