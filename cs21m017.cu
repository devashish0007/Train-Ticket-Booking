#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <iostream>

#define M 25
#define N 100000
#define BLOCKSIZE 1024

using namespace std;


__global__ void init(int * lock, int size)
{
	unsigned id =  blockIdx.x * BLOCKSIZE + threadIdx.x;
	if(id < size)
		lock[id] = 10000;
	
}

__global__ void reserve(int R, int * d_lock, int *d_T, int *d_C, int *d_S, int *d_D, int *d_size, int *d_offset, int volatile *d_seat, int *d_capacity, int * d_class_offset, int *d_R_T, int *d_R_C, int *d_R_S, int *d_R_D, int *d_R_size, int *d_R_seat, int *d_R_result, int start)
{

	int id =  start + blockIdx.x * BLOCKSIZE + threadIdx.x;
	int destination, source, size, train, flag, seat_offset, count, select, execute;
	__shared__ unsigned complete;//, lockvar, round;
//	round = 0;
//	lockvar=0;
	complete = 1;
		
	
	train = d_R_T[id];
	//class_n = d_offset[train] + d_R_C[id];
	seat_offset = d_offset[train] + d_R_C[id] * d_size[train];
	select = d_class_offset[train]+d_R_C[id];
	execute = 1;
	
	flag = 0;
	
	__syncthreads(); // subject to cause deadlock
		
	if(d_R_S[id] > d_R_D[id])
	{
		destination = abs(d_R_S[id] - d_D[train]);
		source = abs(d_R_D[id] - d_D[train]);
		size = abs(source - destination);						    			
	}
	else 
	{
		source = abs(d_R_S[id] - d_S[train]);
		destination = abs(d_R_D[id] - d_S[train]);
		size = abs(source - destination);
	}

		

	/**************************reservation starts*****************************/
		
	do
	{
		complete = 1;
//		if(id ==3)
	//		printf("Round %d\n", round++);
				
		if(execute)
		{
			atomicMin(&(d_lock[select]),id);
		}
		__syncthreads();

		
		if(execute)
		{			
		
			//printf("%d Lock acquired by :%d\n", id);
			if(d_lock[select] == id)
			{
				//printf("Lock acquired by :%d -> %d\n", id,  train);
				// Allot seat
				for(int i = seat_offset + source; i < (seat_offset + destination); i++)
				{
					d_seat[i] = d_seat[i] - d_R_seat[id];
					
					if(d_seat[i] < 0)
					{	
						flag = 1;
						count = i;
						break;						
					}
				}
				
				if(flag)
				{
					// if seat allocation fails
					for(int i = (seat_offset + source); i <= count; i++)
					{
						d_seat[i] += d_R_seat[id];
					}
					d_R_result[id] = 0;
					flag = 0;
				}
				else
				{			
					// if successfull store the result
					d_R_result[id] = size * d_R_seat[id];
				}
				execute = 0;
				//printf("Lock released by :%d", id);
				// unlock the class
				d_lock[select] = 10000;	
				//printf("Lock released by :%d -> %d : %d\n", id,  train, d_R_C[id]);	
			
			}
			else if(execute)
			{
				complete = 0;
			}	
		}
		__syncthreads();
				
	}while(!complete);
	
	__syncthreads();
	/*******************Reservation ends**************************/

}



int main()
{
	int n, count=0, seat_count=0;
	cin >> n;
	
// Allocate memory on cpu
	
    int *T = (int *) malloc ( (n) * sizeof (int) );
    int *C = (int *) malloc ( (n) * sizeof (int) );
    int *S = (int *) malloc ( (n) * sizeof (int) );
    int *D = (int *) malloc ( (n) * sizeof (int) );
    int *size = (int *) malloc ( (n) * sizeof (int) );
    int *offset = (int *) malloc ( (n) * sizeof (int) );
    int *seat = (int *) malloc ( (M * n * 50) * sizeof (int) );
    int *capacity = (int *) malloc ( (M * n) * sizeof (int) ); 
    int * class_offset = (int *) malloc ( (n) * sizeof (int) ); 
      
// Allocate memory on gpu	
	int *d_T, *d_C, *d_S, *d_D, *d_size, *d_offset, *d_seat, *d_capacity, *d_class_offset;

	cudaMalloc(&d_T, (n) * sizeof(int));
	cudaMalloc(&d_C, (n) * sizeof(int));
	cudaMalloc(&d_S, (n) * sizeof(int));
	cudaMalloc(&d_D, (n) * sizeof(int));
	cudaMalloc(&d_size, (n) * sizeof(int));
	cudaMalloc(&d_offset, (n) * sizeof(int));	
	cudaMalloc(&d_capacity, (M * n) * sizeof(int));
	cudaMalloc(&d_class_offset, (M * n) * sizeof(int));
	cudaMalloc(&d_seat, (M * n * 50) * sizeof(int));
	
	for(int i = 0; i < n; i++)
	{
		int t,s,d,c;
		cin >> t >> c >> s >> d;
		T[i] = t;
		S[i] = s;
		D[i] = d;
		C[i] = c;
		offset[i] = seat_count;
		class_offset[i] = count;
		size[i] = abs(s - d);
		for(int j = 0; j < c; j++)
		{
			int c_no, maxc;

			cin >> c_no >> maxc;
			for(int k=seat_count; k < (seat_count + size[i]); k++)
				seat[k] = maxc;
			seat_count += size[i];
			capacity[count] = maxc;
			count++;
		}
	}

	// Copy memory from host to device
	cudaMemcpy(d_T, T, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S, S, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_D, D, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_size, size, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_offset, offset, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_capacity, capacity, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_seat, seat, seat_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_class_offset, class_offset, seat_count * sizeof(int), cudaMemcpyHostToDevice);
	
	// setup lock array	
	
	int *d_lock;
	cudaMalloc(&d_lock, (count) * sizeof(int));
	int initblock = ceil((float(count)/BLOCKSIZE));
	init<<<initblock,BLOCKSIZE>>>(d_lock, count);	


	// Take request input for reservation


	int B, R;
	cin >> B;
	// Allocate memory on cpu
	for(int i = 0; i < B; i++)
	{
		cin>> R;
		
		int *R_T = (int *) malloc ( (R) * sizeof (int) );
		int *R_C = (int *) malloc ( (R) * sizeof (int) );
		int *R_S = (int *) malloc ( (R) * sizeof (int) );
		int *R_D = (int *) malloc ( (R) * sizeof (int) );
		int *R_seat = (int *) malloc ( (R) * sizeof (int) );
		int *R_result = (int *) malloc ( (R) * sizeof (int) );
		int *R_size = (int *) malloc ( (R) * sizeof (int) );

		  
		// Allocate memory on gpu	
		int *d_R_T, *d_R_C, *d_R_S, *d_R_D, *d_R_size, *d_R_seat, *d_R_result, id;

		cudaMalloc(&d_R_T, (R) * sizeof(int));
		cudaMalloc(&d_R_C, (R) * sizeof(int));
		cudaMalloc(&d_R_S, (R) * sizeof(int));
		cudaMalloc(&d_R_D, (R) * sizeof(int));
		cudaMalloc(&d_R_seat, (R) * sizeof(int));
		cudaMalloc(&d_R_result, (R) * sizeof(int));
		cudaMalloc(&d_R_size, (R) * sizeof(int));	

		for(int j = 0; j < R; j++)
		{
			cin>> id>> R_T[j] >> R_C[j] >> R_S[j] >> R_D[j] >> R_seat[j];
			R_size[j] = abs(R_S[j] - R_D[j]);	
		}
		
		// Copy memory from host to device
		cudaMemcpy(d_R_T, R_T, R * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_R_C, R_C, R * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_R_S, R_S, R * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_R_D, R_D, R * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_R_seat, R_seat, R * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_R_size, R_size, R * sizeof(int), cudaMemcpyHostToDevice);
		
		// Organize request.
		int nblocks = R / BLOCKSIZE;
		int extra = R % BLOCKSIZE;
		
		int k = nblocks;
		// Make reservations.
		while(k)
		{	

			reserve<<<1,BLOCKSIZE>>>(R, d_lock, d_T, d_C, d_S, d_D, d_size, d_offset, d_seat, d_capacity, d_class_offset, d_R_T, d_R_C, d_R_S, d_R_D, d_R_size, d_R_seat, d_R_result, BLOCKSIZE*(nblocks - k));
			cudaDeviceSynchronize();
			k--;
		}
		
		reserve<<<1,extra>>>(R, d_lock, d_T, d_C, d_S, d_D, d_size, d_offset, d_seat, d_capacity, d_class_offset, d_R_T, d_R_C, d_R_S, d_R_D, d_R_size, d_R_seat, d_R_result, BLOCKSIZE*nblocks);


		// copy the result back...
		cudaMemcpy(R_result, d_R_result, R * sizeof(int), cudaMemcpyDeviceToHost);
		
		// print result
		long success=0, fail=0, total_seat=0;
		for(int j = 0; j < R; j++)
		{
			if(R_result[j])
			{
				cout<<"success\n";
				success++;
				total_seat += R_result[j];
			}
			else
			{
				cout<<"failure\n";
				fail++;
			}
		}
		
		cout<< success<<" " << fail<<"\n";
		cout<< total_seat <<"\n";
		total_seat = 0;
		free(R_T);
		free(R_C);
		free(R_S);
		free(R_D);
		free(R_seat);
		free(R_result);
	}
	
cudaError_t err = cudaGetLastError();
//printf("error=%d, %s, %s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));

	
	return 0;
}
