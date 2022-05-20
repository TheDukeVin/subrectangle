//
//  gridAgent.cpp
//  gridAgent
//
//  Created by Kevin Du on 5/16/22.
//

#include "subrect.h"

void Grid::inputAgent(Agent* a){
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            a->activation[0][i*M + j] = A[i][j];
        }
    }
}

void Grid::inputSymmetric(Agent* a, int t){
    assert(N == M);
    int m = N-1;
    int sym[8][2][3] = {
        {{ 1, 0, 0},{ 0, 1, 0}},
        {{ 0,-1, m},{ 1, 0, 0}},
        {{-1, 0, m},{ 0,-1, m}},
        {{ 0, 1, 0},{-1, 0, m}},
        {{ 0, 1, 0},{ 1, 0, 0}},
        {{ 1, 0, 0},{ 0,-1, m}},
        {{ 0,-1, m},{-1, 0, m}},
        {{-1, 0, m},{ 0, 1, 0}}
    };
    int x,y;
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            x = sym[t][0][0]*i + sym[t][0][1]*j + sym[t][0][2];
            y = sym[t][1][0]*i + sym[t][1][1]*j + sym[t][1][2];
            a->activation[0][x*M + y] = A[i][j];
        }
    }
}
