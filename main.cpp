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

void setupNet(Agent& net){
    net.initInput(1, 10, 10);
    net.addConvLayer(7, 10, 10, 5, 5);
    net.addConvLayer(7, 10, 10, 5, 5);
    net.addPoolLayer(7, 5, 5);
    net.addDenseLayer(120);
    net.addDenseLayer(1);
    net.quickSetup();
}

void runCycle(double learnRate, int batchSize, double momentum){
    Agent net;
    setupNet(net);
    
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
    setupNet(net);
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
        double agentOutput = G.evalAgent(&net);
        //G.inputAgent(&net);
        //net.pass();
        //double agentOutput = net.output;
        error += squ(expected - agentOutput);
        numCorrect += abs(expected - agentOutput) < 0.5;
    }
    cout<<"Average error: "<<(error / trials)<<'\n';
    cout<<"Accuracy: "<<(numCorrect / trials)<<'\n';
}

void optimize(double learnRate, int batchSize, double momentum){
    const int numNets = 400;
    const int gap = 100;
    Agent nets[numNets];
    for(int i=0; i<numNets; i++){
        setupNet(nets[i]);
    }
    Agent net;
    setupNet(net);
    net.netIn = ifstream("net.in");
    net.readNet();
    net.netIn.close();
    
    Grid G(10, 10);
    
    cout<<"Generating nets...\n";
    
    double error = 0;
    
    for(int t=0; t<numNets*gap; t++){
        for(int i=0; i<batchSize; i++){
            G.randomize(0.5);
            G.inputAgent(&net);
            net.expected = G.byRow();
            net.backProp();
            error += squ(net.expected - net.output);
        }
        net.updateParameters(learnRate / batchSize, momentum);
        if(t%gap == 0){
            cout<<"Net "<<(t / gap)<<" generated with error "<<(error / gap / batchSize) << '\n';
            error = 0;
            net.netOut = ofstream("net.out");
            net.save();
            net.netOut.close();
            nets[t / gap].netIn = ifstream("net.out");
            nets[t / gap].readNet();
            nets[t / gap].netIn.close();
        }
    }
    
    cout<<"First run";
    
    for(int i=0; i<numNets; i++){
        double error = 0;
        for(int j=0; j<100; j++){
            G.randomize(0.5);
            G.inputAgent(&nets[i]);
            nets[i].pass();
            error += squ(nets[i].output - G.byRow());
        }
        cout<<i<<' '<<(error / 100)<<'\n';
    }
    
    cout<<"Confidence method\n";
    
    int count[numNets];
    double errorSum[numNets];
    for(int i=0; i<numNets; i++){
        count[i] = 0;
        errorSum[i] = 0;
    }
    
    for(int N=0; N<numNets * 10000; N++){
        double minVal = 1000;
        int minIndex = -1;
        for(int j=0; j<numNets; j++){
            double newVal = (errorSum[j] - sqrt(N / sqrt(numNets))) / (1 + count[j]);
            if(newVal < minVal){
                minVal = newVal;
                minIndex = j;
            }
        }
        G.randomize(0.5);
        assert(minIndex != -1);
        G.inputAgent(&nets[minIndex]);
        nets[minIndex].pass();
        errorSum[minIndex] += squ(nets[minIndex].output - G.byRow());
        count[minIndex]++;
        if(N%(numNets * 100) == 0){
            cout<<N<<' '<<minIndex<<'\n';
        }
    }
    
    int bestNet = 0;
    double bestAvg = 100;
    for(int i=0; i<numNets; i++){
        assert(count[i] != 0);
        double avg = errorSum[i] / count[i];
        cout<<i<<' '<<count[i]<<' '<<avg<<'\n';
        if(errorSum[i] / count[i] < bestAvg){
            bestAvg = avg;
            bestNet = i;
        }
    }
    nets[bestNet].netOut = ofstream("net.out");
    nets[bestNet].save();
    nets[bestNet].netOut.close();
}

void calculate_standard(){
    int numTrials = 1000000;
    long sum = 0;
    long sumSqu = 0;
    Grid G(10, 10);
    for(int i=0; i<numTrials; i++){
        G.randomize(0.5);
        int val = G.byRow();
        sum += val;
        sumSqu += squ(val);
    }
    double expected = (double) sum / numTrials;
    double expectedSqu = (double) sumSqu / numTrials;
    cout<<"Variance: "<<(expectedSqu - squ(expected))<<'\n';
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
    
    //evaluate();
    
    /*
    double learnRate = 6e-06;
    int batchSize = 30;
    double momentum = 0.9;
    optimize(learnRate, batchSize, momentum);
     */
    
    calculate_standard();
}
