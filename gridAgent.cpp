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
