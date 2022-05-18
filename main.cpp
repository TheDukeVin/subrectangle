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

int main(int argc, const char * argv[]) {
    srand((unsigned) time(NULL));
    /*
    testDeterministic();
     */
    Agent net;
    net.initInput(1, 10, 10);/*
    net.addConvLayer(6, 8, 8, 3, 3);
    net.addConvLayer(6, 8, 8, 3, 3);
    net.addPoolLayer(6, 4, 4);
    net.addDenseLayer(60);
    net.addDenseLayer(1);*/
    net.addConvLayer(4, 10, 10, 3, 3);
    net.addConvLayer(6, 10, 10, 3, 3);
    net.addPoolLayer(6, 5, 5);
    net.addDenseLayer(60);
    net.addDenseLayer(1);
    net.quickSetup();
    //net.readNet();
    
    Grid G(10, 10);
    double error = 0;
    double rate = 6e-06;
    for(int t=0; t<60000; t++){
        for(int i=0; i<batchSize; i++){
            G.randomize(0.5);
            G.inputAgent(&net);
            net.expected = G.byRow();
            net.backProp();
            error += squ(net.output - net.expected);
        }
        if(t>0 && t%1000 == 0){
            double avgError = error / batchSize / 1000;
            cout<<t<<' '<<avgError<<'\n';
            error = 0;/*
            if(avgError < 60 && rate > 1.5e-06){
                rate = 1.5e-06;
                cout<<"Switched to learning rate: "<<rate<<'\n';
            }*/
        }
        net.updateParameters(rate);
    }
    
    net.save();
    net.close();
}
