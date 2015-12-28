# Parallel-K-means-clustering
The k-means clustering algorithm that was parallelized using the openmp library. Have also attached the corresponding project report and the presentation. This project was done as a part of the final project submission for the undergraduate course High Performance Computing.
Compile it using the following command:
	gcc k_means_parallel.c -fopenmp -lm

and then
	./a.out(name of the executable) no_of_rows/datapoints no_of_columns/dimensions no_of_clusters maximum_iterations no_of_threads

	The no_of_threads should be equal to the number of cores you have. It works out the best in that case.
Team:
1. Akshar Varma
2. Yashwant Keswani
