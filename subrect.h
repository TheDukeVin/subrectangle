//
//  subrect.h
//  subrect
//
//  Created by Kevin Du on 5/15/22.
//

#ifndef subrect_h
#define subrect_h

#include "convNet.hpp"

class MinTable{
public:
    int N, L;
    int* nums;
    int** table;
    
    void initTable();
    MinTable(int _N, int* _nums);
    int standardMin(int start, int end);
    int tableMin(int start, int end);
    
    void verify(int maxSize, int numQueries, int numTrials);

private:
    int minIndex(int i1, int i2);
};

class Grid{
public:
    int N, M;
    bool** A;
    
    Grid(int _N, int _M);
    Grid(int _N, int _M, vector<bool> vals);
    int checkAll();
    int DP();
    int byRow();
    void randomize(double density);
    void verifyAllDP(int maxSize, double density, int numTrials);
    void verifyDPRow(int maxSize, double density, int numTrials);
    
    // Network interaction
    
    void inputAgent(Agent* a);
    
private:
    MinTable* T;
    int fromSizes(int start, int end);
};

#endif /* subrect_h */
