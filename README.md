# Earth Movers Distance code
 
 This program is an exploration into a revealed issue with the LA County BMP Performance Index framework.  
 
 The issue is in consolidating a set of categorical data into a single numerical value.  Averages work, but there become many sets that can have the same average, and we want a way to differentiate between sets that are qualitatively very different.
 
 The solution I have scripted here is using an algorithm called Earth Mover's Distance.  Essentially, this algorithm is quantifying the difference between two histograms - which fits well with our categorical data.
 
 The tricky bit is designing the "Distance Matrix", or the amount of "work" that the algorithm thinks is necessary to transform one histogram into another.  I understand now the way to construct a distance matrix, but the way I have decided to do so is partially arbitrary.
 
 Using integer rankings between the categories as the "distances" between categories DOES NOT remove the issue where very different sets qualitatively result in the same quanitative metric.  What you need is an asymmetric distance matrix (not in the primary direction, but orthogonally), and that can be achieved by considering the distance between categories as the number of "Threshold Lines" that need to be crossed.  A threshold line, here, is defined as one of: Influent = Threshold, Effluent = Threshold, and Influent = Effluent.
 
 The resulting distance matrix between the categories looks like the following:
 
 Categories:
 - Failing
 - Subpar
 - Contributing
 - Surpassing
 - Succeeding

Distance Matrix:

|0  1   2   3   4|

|1  0   4   2   3|

|2  4   0   2   1|

|3  2   2   0   1|

|4  3   1   1   0|


Earth Mover's Distances are naturally an increasing metric, i.e., larger numbers indicate worse histogram agreement.  To allow for EMDs to be compared directly against quintile or average scores, I created a normalization equation that preserves the shape of the EMD distribution, but scales it as 0 (Worst) - 5 (Best)

normalized EMD = max Average/Quintile score (5) - max Average/Quintile score * EMD / max EMD
