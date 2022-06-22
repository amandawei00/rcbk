#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void spline(double *x, double *y, double *b, double *c, double *d, int n){
	int clyde;
	int gap = n - 1;
	double h;

	/* check input */
	if (n < 2) {
	    return;
	}
	if (n < 3) {
	    b[0] = (y[1]-y[0])/(x[1]-x[0]); //linear interpolation
	    c[0] = 0;
	    d[0] = 0;

	    b[1] = b[0];
	    c[1] = 0;
	    d[1] = 0;

	    return;
	}

	/* step 1: preparation */
	d[0] = x[1] - x[0];
	c[1] = (y[1] - y[0])/(d[0]);
	for (int i=1; i < gap; i++){
	    d[i] = x[i+1] - x[i];
	    b[i] = 2.0 * (d[i-1] + d[i]);
	    c[i+1] = (y[i+1] - y[i])/(d[i]);
	    c[i] = c[i+1] - c[i];
	}	

	/* step 2: end conditions */
	b[0] = -d[0];
	b[n-1] = -d[n-2];
	c[0] = 0.0;
	c[n-1] = 0.0;

	if (n != 3){
            c[0] = c[2]/(x[3]-x[1]) - c[1]/(x[2] - x[0]);
	    c[n-1] = c[n-2]/(x[n-1] - x[n-3]) - c[n-3]/(x[n-2] - x[n-4]);
	    c[0] = (c[0]) * (d[0]) * (d[0])/(x[3] - x[0]);
	    c[n-1] = -(c[n-1]) * (d[n-2]) * (d[n-2]) / (x[n-1] - x[n-4]);
	}

	/* step 3: forward elimination */
	for (int i=1; i < n; i++){
            h = d[i-1]/(b[i-1]);
	    b[i] = b[i] - h * (d[i-1]);
	    c[i] = c[i] - h * (c[i-1]);
	}	    

	/* step 4: back substitution */
	c[n-1] = c[n-1]/b[n-1];
	for (int j=0; j < gap; j++){
	    clyde = n - j - 2;
	    c[clyde] = (c[clyde] - d[clyde]*c[clyde+1])/b[clyde];
	}

	/* step 5: compute spline coefficients */
	b[n-1] = (y[n-1] - y[gap-1])/(d[gap-1]) + d[gap-1]*(c[gap-1] + 2.0*c[n-1]);  //please check this
	for(int i=0; i < gap; i++){
	    b[i] = (y[i+1] - y[i])/(d[i]) - d[i]*(c[i+1] + 2.0*(c[i]));
	    d[i] = (c[i+1] - c[i])/(d[i]);
	    c[i] = 3.0 * (c[i]);
	}
	c[n-1] = 3.0 * c[n-1];
	d[n-1] = d[n-2];
}


double ispline(double u, double *x, double *y, double *b, double *c, double *d, int n){
	double result;
	double dx;
	int i, j, k;
	/* if u is outside x() interval, take boundary value (left or right) */
	//printf("ispline entered");
        if (u <= x[0]){
	    return y[0];
	} else if (u >= x[n-1]) {
	    return y[n-1];
	}	
        //printf("if statement complete");
	/* binary search for i, s.t. x[i] <= u <= x[i+1]*/
	i = 0;
	j = n;

	do{
	    k = (i + j)/2;
	    if (u < x[k]) {
		j = k;
	    } else {
		i = k;
	    }
	} while (j > i+1);
        //printf("do statement complete");
	dx = u - x[i];
	result = y[i] + dx * (b[i] + dx * (c[i] + dx * (d[i])));
        //printf("ispline complete");
	return result;
}


double * linspace(double a, double b, int c){
	double *line = malloc(c * sizeof(double));
	double delta = (b - a)/(c-1);
	
	for(int i=0; i<c-1; i++){
		line[i] = a + delta * i;
	}
	
	line[c-1] = b;
	return line;
}


int main(){
	printf("COOLBEANS :) \n");
	
	int n = 11;
	int m = 21;

	double xmin = 0.0;
	double xmax = 2.0;

	double *x = linspace(xmin, xmax, n);
	double *x_grid = linspace(xmin, xmax, m);
	double *y = malloc(n * sizeof(double));

	for (int i=0; i<n; i++){
		y[i] = sin(x[i]);
	}

	double *b = malloc(n * sizeof(double));
	double *c = malloc(n * sizeof(double));
	double *d = malloc(n * sizeof(double));

	spline(x, y, b, c, d, n);

	double *y_grid = malloc(m * sizeof(double));
	for (int i=0; i<m; i++){
		y_grid[i] = ispline(x_grid[i], x, y, b, c, d, n);
		printf("%f ", x_grid[i]);
		printf(", %f ", y_grid[i]);
		printf("\n");
	}

	printf("done");

        free(x);
        free(y);
	free(x_grid);
	free(y_grid);

	return(0);
}
