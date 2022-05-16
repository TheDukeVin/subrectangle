//
//  main.cpp
//  subrectangle
//
//  Created by Kevin Du on 5/15/22.
//

#include "subrect.h"

int main(int argc, const char * argv[]) {
    srand((unsigned) time(NULL));
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
    /*
    int nums[11] = {5, 8, 2, 7, 10, 1, 4, 2, 8, 5, 9};
    MinTable M(11, nums);
    M.verify(100, 100, 10);
     */
    
}
