#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

#define DEBUG_MSE 0
/*The parallel version of eucledian which was used.


double eucledian_distance(double *p1,double *p2,int n){
	double ans=0;
    #pragma omp parallel reduction(+:ans)
    {
		int who_am_i=omp_get_thread_num();
		int total=omp_get_num_threads();		
		int start=(int)(n*who_am_i/(double)(total));
		who_am_i++;		
		int end=(int)(n*(who_am_i)/(double)(total));
		if(who_am_i==total)
			end=n;
		who_am_i--;
		double sum=0;
		int i;
		for(i=start;i<end;i++)
			sum+=((p1[i]-p2[i])*(p1[i]-p2[i]));
		ans+=sum;  	
	}

return sqrt(ans);
}


*/

double eucledian_distance(double *p1,double *p2,int n){
  double sum=0;
  int i;
  for(i=0;i<n;i++)
    sum+=((p1[i]-p2[i])*(p1[i]-p2[i]));
  return sqrt(sum);
}

int rand_lim(int limit){
  /* return a random number between 0 and limit inclusive.*/
  int divisor = RAND_MAX/(limit+1);
  int retval;
  do{
    retval = rand() / divisor;
  }while (retval > limit);
  
  return retval;
}

double *calculate_mean(double **data_points,int no_of_rows,int no_of_columns){
  int i,j;
  double *mean=(double *)malloc(no_of_columns*sizeof(double));	  
  for(i=0;i<no_of_rows;i++){
    for(j=0;j<no_of_columns;j++){
      mean[j]+=data_points[i][j];
    }
  }
	for(j=0;j<no_of_columns;j++)
		mean[j]/=(double)(no_of_rows);
  return mean;
}

double *calculate_standard_deviation(double **data_points,double *mean,int no_of_rows,int no_of_columns){
  int i,j;
  double *std_dev=(double *)malloc(no_of_columns*sizeof(double));
  for(i=0;i<no_of_rows;i++){
    double sum=0.0;
    for(j=0;j<no_of_columns;j++){
      std_dev[j]+=((data_points[i][j]-mean[j])*(data_points[i][j]-mean[j]));
    }
  }
  for(j=0;j<no_of_columns;j++)
	std_dev[j]=sqrt(std_dev[j]/(double)(no_of_columns));

  return std_dev;
}

double **normalize_datapoints(double **data_points,double *mean,double *std_dev,int no_of_rows,int no_of_columns){
  int i,j;
  double **data_points_normalized=(double **)malloc(no_of_rows*sizeof(double*));
  for(i=0;i<no_of_rows;i++){
    data_points_normalized[i]=(double *)malloc(no_of_columns*sizeof(double));
    for(j=0;j<no_of_columns;j++)
      data_points_normalized[i][j]=((data_points[i][j]-mean[j])/std_dev[j]);
  }
  return data_points_normalized;
}

int *allocate_initial_centroids(double **data_points_normalized, int no_of_rows,int no_of_clusters){
  //http://stackoverflow.com/questions/1608181/unique-random-numbers-in-an-integer-array-in-the-c-programming-language
  int *indices=(int *)malloc(no_of_clusters*sizeof(int));
  int M=no_of_clusters, N=no_of_rows;
  int in, im=0;
  for (in = 0; in < N && im < M; ++in) {
    int rn = N - in;
    int rm = M - im;
    if (rand() % rn < rm)    
      indices[im++] = in;
  }
  return indices;
}

int *cluster(double **data_points_normalized,int *initial_centroids,int no_of_rows,int no_of_columns,int no_of_clusters, int max_iterations){
  int i,j;
  int *labels=(int *)malloc(no_of_rows*sizeof(int));
  int *points_in_each_cluster=(int *)malloc(no_of_clusters*sizeof(int));
  points_in_each_cluster[0]=no_of_rows;
  for(i=1;i<no_of_clusters;i++)
    points_in_each_cluster[i]=0;
  double **centroids=(double **)malloc(no_of_clusters*sizeof(double *));
  for(i=0;i<no_of_rows;i++)
    labels[i]=0;
  for(i=0;i<no_of_clusters;i++){
    centroids[i]=(double *)malloc(no_of_columns*sizeof(double));
    for(j=0;j<no_of_columns;j++){
      centroids[i][j]=data_points_normalized[initial_centroids[i]][j];
    }
  }

  int re_assignments=100;
  double error;
  int count=1;
  while(re_assignments!=0 && max_iterations>0){
    double start = omp_get_wtime();
    clock_t cpu_start = clock();
    re_assignments=0;
    /* printf("Iteration started.\n"); */
    //Assigning points to each centroid.
	count++;
	int **temp_labels;
	#pragma omp parallel 
	{
		int total=omp_get_num_threads();
		#pragma omp single
		{
			temp_labels=(int **)malloc(total*sizeof(int(*)));		
		}
		//printf("Entered parallel region\n");
				
		
		int a,b;
		int n_rows=(int)(no_of_rows/(double)(total));			
		for(a=0;a<total;a++)
			temp_labels[a]=(int *)malloc(n_rows*sizeof(int));
		int who_am_i=omp_get_thread_num();
		int start=(int)(no_of_rows*who_am_i/(double)(total));
		who_am_i++;		
		int end=(int)(no_of_rows*(who_am_i)/(double)(total));
		if(who_am_i==total)
			end=no_of_rows;
		who_am_i--;		
		//printf("%d %d\n",start,end);
		for(a=start;a<end;a++)
		{
			int allocated_to=1234567;
			int previous_to=labels[a];
			double error_for_this_point=9999999999;
			for(b=0;b<no_of_clusters;b++)
			{
				double fook=(eucledian_distance(data_points_normalized[a],centroids[b],no_of_columns));
				if(fook<error_for_this_point)
				{
					//printf("Label %d %d %d %d\n",a,b,allocated_to,previous_to);					
					error_for_this_point=fook;
	  				allocated_to=b;
				}
 		    }				
      		if(previous_to!=allocated_to)
			{
				#pragma omp atomic
				re_assignments++;
			}
      		temp_labels[who_am_i][a-start]=allocated_to;
		}
		#pragma omp barrier
		#pragma omp single
		{
			max_iterations-=1;
			for(i=0;i<no_of_rows;i++)
			{
				labels[i]=temp_labels[i/n_rows][i%n_rows];		
			}			
		}
	}
    /* printf("Points assigned to centroids.\n"); */
    //Recalculating the centroids.
	for(i=0;i<no_of_clusters;i++)
		points_in_each_cluster[i]=0;	
	for(i=0;i<no_of_rows;i++)
		points_in_each_cluster[labels[i]]+=1;
    for(i=0;i<no_of_clusters;i++){
      for(j=0;j<no_of_columns;j++)
		centroids[i][j]=0.00;
    }

    for(i=0;i<no_of_rows;i++){
      for(j=0;j<no_of_columns;j++)
	centroids[labels[i]][j]+=data_points_normalized[i][j];
    }
      
    for(i=0;i<no_of_clusters;i++){
      for(j=0;j<no_of_columns;j++)
	centroids[i][j]/=points_in_each_cluster[i];
    }
    /* printf("Centroids recalculated.\n"); */
    printf("Iteration Complete; Error=%lf; re_assignments=%d; Clock=%.3f\n",error,re_assignments,(omp_get_wtime() - start));
    /* printf("Clock: %.2f\n", (double)(omp_get_wtime() - start)); */
    /* printf("CPU_time: %.2f.\n", (clock() - start)/(double)CLOCKS_PER_SEC); */
  }

  return labels;
}

double mse(double **data_points_normalized, int *labels,int no_of_rows, int no_of_columns,int no_of_clusters){
  double error=0;
  int i,j,k;
  double **centroids=(double **)malloc(no_of_clusters*sizeof(double));
  int *points_in_each_cluster=(int *)malloc(no_of_clusters*sizeof(int));
  DEBUG_MSE && printf("made something\n");
  for(i=0;i<no_of_clusters;i++){
    centroids[i]=(double *)malloc(no_of_rows*sizeof(double));
    for(j=0;j<no_of_columns;j++){
      centroids[i][j]=0.0;
    }
    points_in_each_cluster[i]=0;
  }
  DEBUG_MSE && printf("made everything\n");
  for(i=0;i<no_of_rows;i++){
    points_in_each_cluster[labels[i]]++;
    for(j=0;j<no_of_columns;j++){
      centroids[labels[i]][j]+=data_points_normalized[i][j];
    }
  }
  for(i=0;i<no_of_clusters;i++){
    for(j=0;j<no_of_columns;j++)
      centroids[i][j]/=points_in_each_cluster[i];
  }
  DEBUG_MSE && printf("init done\n");  
  for(i=0;i<no_of_rows;i++)
    for(j=0;j<no_of_columns;j++)
      error+=(data_points_normalized[i][j]-centroids[labels[i]][j])*(data_points_normalized[i][j]-centroids[labels[i]][j]);
  
  error=error/no_of_rows;
  
  return error;
}

double dunns_index(double **data_points_normalized, int *labels,int no_of_rows, int no_of_columns,int no_of_clusters){
  int i,j,k;
  double **centroids=(double **)malloc(no_of_clusters*sizeof(double));
  int *points_in_each_cluster=(int *)malloc(no_of_clusters*sizeof(int));
	
  for(i=0;i<no_of_clusters;i++){
    centroids[i]=(double *)malloc(no_of_rows*sizeof(double));
    for(j=0;j<no_of_columns;j++){
      centroids[i][j]=0.0;
    }
    points_in_each_cluster[i]=0;
  }

  for(i=0;i<no_of_rows;i++){
    points_in_each_cluster[labels[i]]++;
    for(j=0;j<no_of_columns;j++){
      centroids[labels[i]][j]+=data_points_normalized[i][j];
    }
  }
  for(i=0;i<no_of_clusters;i++){
    for(j=0;j<no_of_columns;j++)
      centroids[i][j]/=points_in_each_cluster[i];
  }
  double **inter=(double**)malloc(no_of_clusters*sizeof(double*));
  for(i=0;i<no_of_clusters;i++)
    inter[i]=(double *)malloc(no_of_clusters*sizeof(double));
  double minInter=999999999;
  
  double *intra=(double*)malloc(no_of_clusters*sizeof(double));
  for(i=0;i<no_of_clusters;i++){
    intra[i]=-999.0000;
  }
  double maxIntra=-99999999;

  for(i=0;i<no_of_clusters;i++){
    for(j=0;j<i;j++){
      inter[i][j]=eucledian_distance(centroids[i],centroids[j],no_of_columns);
      if(inter[i][j]!=0 && inter[i][j]<minInter)
	minInter=inter[i][j];
    }
  }
   
  for(i=0;i<no_of_clusters;i++){
    for(j=0;j<no_of_rows;j++){
      if(labels[j]==i){
	for(k=j+1;k<no_of_rows;k++){
	  if(labels[k]==i){
	    if(eucledian_distance(data_points_normalized[j], data_points_normalized[k], no_of_columns)>intra[i]){
	      intra[i]=eucledian_distance(data_points_normalized[j], data_points_normalized[k], no_of_columns);
	    }
	  }
	}
      }
    }
    // if(eucledian_distance(centroids[labels[i]], data_points[i], dimensions)>intra[labels[i]])
    //   intra[labels[i]]=eucledian_distance(centroids[labels[i]], data_points[i], dimensions);
    if(intra[i]!=0 && maxIntra<intra[i])
      maxIntra=intra[i];
  }

  return (double)minInter/(double)maxIntra;

}

void write_to_file(char *filename,int *labels,int no_of_rows){
  FILE *f=fopen(filename,"w");
  int i;
  for(i=0;i<(no_of_rows-1);i++)
    fprintf(f,"%d,",labels[i]+1);
  fprintf(f,"%d",labels[no_of_rows-1]+1);
}

int main(int argc, char* argv[]){
  int i,j; //Iteration variables.
  int n=atoi(argv[1]);//No of points in the data set.
  int dimensions=atoi(argv[2]);//Dimension of each point.
  int max_iterations=atoi(argv[4]); //The maximum number of iterations.
  omp_set_num_threads(atoi(argv[5]));

  double **data_points=(double **)malloc(n*sizeof(double*));
  double *mean=(double *)malloc(dimensions*sizeof(double));
  double *std_dev=(double *)malloc(dimensions*sizeof(double));
  // double maxDI=-22;
  // int maxDIindex;

  int seed=time(NULL);
  printf("Seed: %d\n", seed);
  srand(seed);

  double **data_points_normalized=(double **)malloc(n*sizeof(double*));
  for(i=0;i<n;i++){
    data_points_normalized[i]=(double *)malloc(dimensions*sizeof(double));
  }

  double start=omp_get_wtime();
  printf("Reading file.\n");
  FILE *input = fopen("HiPC2015_IntelData_40k.csv", "r");
  for(i=0;i<n;i++){
    data_points[i]=(double *)malloc(dimensions*sizeof(double));
    for(j=0;j<dimensions;j++){
      fscanf(input,"%lf,",&data_points[i][j]);
    }
  }
  //Calculate the mean
  printf("Done. %f\n",omp_get_wtime()-start);
  start=omp_get_wtime();
  printf("Calculating mean.\n");
  mean=calculate_mean(data_points,n,dimensions);
  printf("Done.\n");
  printf("Calculating standard deviation.\n");
  std_dev=calculate_standard_deviation(data_points,mean,n,dimensions);
  printf("Done.\n");
  printf("Normalizing.\n");
  data_points_normalized=normalize_datapoints(data_points,mean,std_dev,n,dimensions);
  printf("Done.\n"); 


  int no_of_clusters=atoi(argv[3]);
  int *initial_centroids=(int *)malloc(no_of_clusters*sizeof(int));
  initial_centroids=allocate_initial_centroids(data_points_normalized,n,no_of_clusters);
  int *labels=(int *)malloc(n*sizeof(int));
  printf("Clustering.\n");
  labels=cluster(data_points_normalized,initial_centroids,n,dimensions,no_of_clusters,max_iterations);
  printf("Done.\n");
  printf("Everything in: %.3f\n",omp_get_wtime()-start);
  start=omp_get_wtime();
  printf("Calculating Dunn's Index.\n");
  //double res=dunns_index(data_points_normalized, labels,n, dimensions,no_of_clusters);
  //printf("Dunn=%lf. Time=%.3f\n",res,omp_get_wtime()-start);
  printf("MSE=%.3f\n",mse(data_points_normalized,labels,n,dimensions,no_of_clusters));
  char* out_file=(char*)malloc(50*sizeof(char));
  sprintf(out_file,"labels_%d_%d2.csv", no_of_clusters, seed);
  write_to_file(out_file,labels,n);
  return 0;
}

