# subrectangle
Comparing deterministic and neural network based algorithms for counting sub rectangles of a grid.

Given a grid of black and white cells, how can we count the number of black rectangles formed by these black cells?

BWW\
BBB

For instance, in the grid above, B represents black cells and W represents white cells. In that example, there are eight black rectangles: four 1x1, one 2x1, two 1x2, and one 1x3 rectangle. How do we count the number of rectangles algorithmically? There are many deterministic algorithms to solve this problem, but this project investigates using a neural network to solve this problem.

On a 10x10 grid where each cell is black with 0.5 probability, I trained a convolutional neural network to calculate this number. The results show significant progress as well as much to be desied. My best network had a variance in its output of around 0.8. Given that the data has a variance of around xxx, this is a very small error, but it only gives the correct answer around 54% of the time. Due to the diminishing returns on training larger, more accurate networks, I decided this is a good place to stop for this project.
