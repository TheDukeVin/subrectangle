//
//  mintable.cpp
//  mintable
//
//  Created by Kevin Du on 5/15/22.
//

#include "subrect.h"

MinTable::MinTable(int _N, int* _nums){
    N = _N;
    nums = _nums;
    initTable();
}

int MinTable::standardMin(int start, int end){
    int minIndex = start;
    for(int i=start+1; i<=end; i++){
        if(nums[minIndex] > nums[i]){
            minIndex = i;
        }
    }
    return minIndex;
}

int MinTable::tableMin(int start, int end){
    int l = -1;
    int size = end - start + 1;
    while(size != 0){
        size >>= 1;
        l++;
    }
    return minIndex(table[l][start], table[l][end - (1 << l) + 1]);
}

int MinTable::minIndex(int i1, int i2){
    if(nums[i1] < nums[i2]) return i1;
    return i2;
}

void MinTable::initTable(){
    L = 0;
    int _N = N;
    while(_N != 0){
        _N >>= 1;
        L++;
    }
    table = new int*[L];
    table[0] = new int[N];
    for(int i=0; i<N; i++) table[0][i] = i;
    for(int d=1; d<L; d++){
        int sub = (1 << d) - 1;
        int gap = 1 << (d - 1);
        table[d] = new int[N - sub];
        for(int i=0; i<N - sub; i++){
            table[d][i] = minIndex(table[d-1][i], table[d-1][i + gap]);
        }
    }
}

void MinTable::verify(int maxSize, int numQueries, int numTrials){
    for(int t=0; t<numTrials; t++){
        N = rand() % maxSize + 1;
        nums = new int[N];
        for(int i=0; i<N; i++) nums[i] = rand();
        initTable();
        for(int i=0; i<numQueries; i++){
            int start = 1;
            int end = 0;
            while(start > end){
                start = rand() % N;
                end = rand() % N;
            }
            int smin = standardMin(start, end);
            int tmin = tableMin(start, end);
            assert(smin == tmin);
        }
    }
}
