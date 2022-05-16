//
//  grid.cpp
//  grid
//
//  Created by Kevin Du on 5/15/22.
//

#include "subrect.h"

Grid::Grid(int _N, int _M, vector<bool> vals){
    N = _N;
    M = _M;
    A = new bool*[N];
    for(int i=0; i<N; i++){
        A[i] = new bool[M];
        for(int j=0; j<M; j++){
            A[i][j] = vals[i*M + j];
        }
    }
}

int Grid::checkAll(){
    int counter = 0;
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            for(int k=i; k<N; k++){
                for(int l=j; l<M; l++){
                    bool isRect = true;
                    for(int x=i; x<=k; x++){
                        for(int y=j; y<=l; y++){
                            if(!A[x][y]) isRect = false;
                        }
                    }
                    if(isRect) counter++;
                }
            }
        }
    }
    return counter;
}

int Grid::DP(){
    bool isRect[N][M][N][M];
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            for(int k=i; k<N; k++){
                for(int l=j; l<M; l++){
                    isRect[i][j][k][l] = false;
                }
            }
        }
    }
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            isRect[i][j][i][j] = A[i][j];
        }
    }
    for(int r=0; r<N; r++){
        for(int c=0; c<M; c++){
            if(r == 0 && c == 0) continue;
            for(int i=0; i<N-r; i++){
                for(int j=0; j<M-c; j++){
                    if(r == 0) isRect[i][j][i+r][j+c] = isRect[i][j][i+r][j+c-1] && isRect[i][j+1][i+r][j+c];
                    else isRect[i][j][i+r][j+c] = isRect[i][j][i+r-1][j+c] && isRect[i+1][j][i+r][j+c];
                }
            }
        }
    }
    int counter = 0;
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            for(int k=i; k<N; k++){
                for(int l=j; l<M; l++){
                    if(isRect[i][j][k][l]){
                        counter++;
                    }
                }
            }
        }
    }
    return counter;
}

void Grid::verifyAllDP(int maxSize, double density, int numTrials){
    for(int t=0; t<numTrials; t++){
        delete A;
        N = rand() % maxSize + 1;
        M = rand() % maxSize + 1;
        A = new bool*[N];
        for(int i=0; i<N; i++){
            A[i] = new bool[M];
            for(int j=0; j<M; j++){
                A[i][j] = rand() < density * RAND_MAX;
            }
        }
        assert(checkAll() == DP());
    }
}

int Grid::byRow(){
    int sizes[N];
    for(int i=0; i<N; i++) sizes[i] = 0;
    T = new MinTable(N, sizes);
    int sum = 0;
    for(int j=0; j<M; j++){
        for(int i=0; i<N; i++){
            if(sizes[i] == 0){
                int k = j;
                while(k<M && A[i][k]) k++;
                sizes[i] = k - j;
            }
            else sizes[i]--;
        }
        T->initTable();
        sum += fromSizes(0, N-1);
    }
    return sum;
}

int Grid::fromSizes(int start, int end){
    if(start > end) return 0;
    int minIndex = T->tableMin(start, end);
    int leftSum = fromSizes(start, minIndex - 1);
    int rightSum = fromSizes(minIndex + 1, end);
    int includeCount = (minIndex - start + 1) * (end - minIndex + 1) * T->nums[minIndex];
    return leftSum + rightSum + includeCount;
}

void Grid::verifyDPRow(int maxSize, double density, int numTrials){
    for(int t=0; t<numTrials; t++){
        delete A;
        N = rand() % maxSize + 1;
        M = rand() % maxSize + 1;
        A = new bool*[N];
        for(int i=0; i<N; i++){
            A[i] = new bool[M];
            for(int j=0; j<M; j++){
                A[i][j] = rand() < density * RAND_MAX;
            }
        }
        assert(DP() == byRow());
    }
}
