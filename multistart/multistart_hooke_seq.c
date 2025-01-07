#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define MAXVARS (250) /* max # of variables */
#define RHO_BEGIN (0.5) /* stepsize geometric shrink */
#define EPSMIN (1E-6) /* ending value of stepsize */
#define IMAX (5000) /* max # of iterations */

/* global variables */
unsigned long funevals = 0;

/* Rosenbrocks classic parabolic valley ("banana") function */
double f(double *x, int n) {
    double fv;
    int i;

    funevals++;
    fv = 0.0;
    #pragma omp for
    for (i = 0; i < n - 1; i++) /* rosenbrock */
        fv = fv + 100.0 * pow((x[i + 1] - x[i] * x[i]), 2) + pow((x[i] - 1.0), 2);

    return fv;
}

/* given a point, look for a better one nearby, one coord at a time */
double best_nearby(double delta[MAXVARS], double point[MAXVARS], double prevbest, int nvars) {
    double z[MAXVARS];
    double minf, ftmp;
    int i;
    minf = prevbest;
    for (i = 0; i < nvars; i++)
        z[i] = point[i];
    for (i = 0; i < nvars; i++) {
        z[i] = point[i] + delta[i];
        ftmp = f(z, nvars);
        if (ftmp < minf)
            minf = ftmp;
        else {
            delta[i] = 0.0 - delta[i];
            z[i] = point[i] + delta[i];
            ftmp = f(z, nvars);
            if (ftmp < minf)
                minf = ftmp;
            else
                z[i] = point[i];
        }
    }
    for (i = 0; i < nvars; i++)
        point[i] = z[i];

    return (minf);
}

int hooke(int nvars, double startpt[MAXVARS], double endpt[MAXVARS], double rho, double epsilon, int itermax) {
    double delta[MAXVARS];
    double newf, fbefore, steplength, tmp;
    double xbefore[MAXVARS], newx[MAXVARS];
    int i, keep;
    int iters, iadj;

    for (i = 0; i < nvars; i++) {
        newx[i] = xbefore[i] = startpt[i];
        delta[i] = fabs(startpt[i] * rho);
        if (delta[i] == 0.0)
            delta[i] = rho;
    }
    iadj = 0;
    steplength = rho;
    iters = 0;
    fbefore = f(newx, nvars);
    newf = fbefore;
    while ((iters < itermax) && (steplength > epsilon)) {
        iters++;
        iadj++;

        for (i = 0; i < nvars; i++) {
            newx[i] = xbefore[i];
        }
        newf = best_nearby(delta, newx, fbefore, nvars);

        keep = 1;
        while ((newf < fbefore) && (keep == 1)) {
            iadj = 0;
            for (i = 0; i < nvars; i++) {
                if (newx[i] <= xbefore[i])
                    delta[i] = 0.0 - fabs(delta[i]);
                else
                    delta[i] = fabs(delta[i]);
                tmp = xbefore[i];
                xbefore[i] = newx[i];
                newx[i] = newx[i] + newx[i] - tmp;
            }
            fbefore = newf;
            newf = best_nearby(delta, newx, fbefore, nvars);
            if (newf >= fbefore)
                break;

            keep = 0;
            for (i = 0; i < nvars; i++) {
                keep = 1;
                if (fabs(newx[i] - xbefore[i]) > (0.5 * fabs(delta[i])))
                    break;
                else
                    keep = 0;
            }
        }
        if ((steplength >= epsilon) && (newf >= fbefore)) {
            steplength = steplength * rho;
            for (i = 0; i < nvars; i++) {
                delta[i] *= rho;
            }
        }
    }
    for (i = 0; i < nvars; i++)
        endpt[i] = xbefore[i];

    return (iters);
}

double get_wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

int main(int argc, char *argv[]) {
    double startpt[MAXVARS], endpt[MAXVARS];
    int itermax = IMAX;
    double rho = RHO_BEGIN;
    double epsilon = EPSMIN;
    int nvars;
    int trial, ntrials;
    double fx;
    int i, jj;
    double t0, t1;
    double global_best_fx = 1e10;
    double global_best_pt[MAXVARS];
    int global_best_trial = -1;
    int global_best_jj = -1;

    ntrials = 16*1024;    /* number of trials */
    nvars = 16;           /* number of variables (problem dimension) */
    srand48(24);   /* srand48(time(0)); */

    t0 = get_wtime();
    
    #pragma omp parallel private(i, jj, fx, startpt, endpt)
    {
        double best_fx = 1e10;
        double best_pt[MAXVARS];
        int best_trial = -1;
        int best_jj = -1;

        for (int j = 0; j < MAXVARS; j++) best_pt[j] = 0.0;

        #pragma omp for nowait
        for (trial = 0; trial < ntrials; trial++) {
            /* starting guess for rosenbrock test function, search space in [-4, 4) */
            for (i = 0; i < nvars; i++) {
                startpt[i] = 8.0 * drand48() - 4.0;
            }

            jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax);
            #if DEBUG
            printf("\n\n\nHOOKE %d USED %d ITERATIONS, AND RETURNED\n", trial, jj);
            for (i = 0; i < nvars; i++)
                printf("x[%3d] = %15.7le \n", i, endpt[i]);
            #endif

            fx = f(endpt, nvars);
            #if DEBUG
            printf("f(x) = %15.7le\n", fx);
            #endif
            
            if (fx < best_fx) {
                best_trial = trial;
                best_jj = jj;
                best_fx = fx;
                for (i = 0; i < nvars; i++) {
                    best_pt[i] = endpt[i];
                }
            }
        }

        #pragma omp critical
        {
            if (best_fx < global_best_fx) {
                global_best_trial = best_trial;
                global_best_jj = best_jj;
                global_best_fx = best_fx;
                for (i = 0; i < nvars; i++) {
                    global_best_pt[i] = best_pt[i];
                }
            }
        }
    }
    
    t1 = get_wtime();

    printf("\n\nFINAL RESULTS:\n");
    printf("Elapsed time = %.3lf s\n", t1-t0);
    printf("Total number of trials = %d\n", ntrials);
    printf("Total number of function evaluations = %ld\n", funevals);
    printf("Best result at trial %d used %d iterations, and returned\n", global_best_trial, global_best_jj);
    for (i = 0; i < nvars; i++) {
        printf("x[%3d] = %15.7le \n", i, global_best_pt[i]);
    }
    printf("f(x) = %15.7le\n", global_best_fx);

    return 0;
}

