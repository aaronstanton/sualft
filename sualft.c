/* Copyright (c) Signal Analysis and Imaging Group (SAIG), University of Alberta, 2013.*/
/* All rights reserved.                       */
/* sualft  :  $Date: May 2013- Last version May 2013  */

#include "su.h"
#include "cwp.h"
#include "segy.h"
#include "header.h"
#include <time.h>
#include "fftw3.h" 
#include "alft.h"

#ifndef MARK
#define MARK fprintf(stderr,"%s @ %u\n",__FILE__,__LINE__);fflush(stderr);
#endif

/*********************** self documentation **********************/
char *sdoc[] = {
  " 	   							                                  ",
  " SUALFT  2D regularization using the Anti-Leakage Fourier Transform ",
  "         regularization is keyed on the offset header word          ",
  "                                                                   ",
  " User provides:                                                    ",
  "           < in.su, > out.su                                       ",
  " Other parameters: (parameter and its default setting)             ",
  "                                                                   ",
  "           verbose=0; (=1 to show messages)                        ",
  "           nx=10000; (number of input traces)                      ",
  "           dh=10 (desired output spatial sampling (meters))        ",
  "           Ltw=200; (length of time window in samples)             ",
  "           Dtw=10; (overlap of time windows in samples)            ",
  "           padt=2; (padding factor in time dimension)              ",
  "           padx=2; (padding factor in spatial dimension)           ",
  "           fmin=0; (min frequency to process: careful, no taper!)  ",
  "           fmax=0.5/dt; (max frequency to process: careful, no taper!)",
  "           niter=100; (number of iterations of ALFT)               ",
  "                                                                   ",
  " Example:                                                          ",
  " # make synthetic data consisting of two linear events             ",
  "           suwaveform type=ricker1 fpeak=30 | sugain pbal=1 > wavelet.su",
  "           suplane nt=200 npl=2 ntr=500 dip1=0 dip2=1 len1=500 len2=500 | suconv sufile=wavelet.su > tmp1.su",
  " # set offset header to 1m sampling                                 ",
  "           suchw key1=offset key2=tracl a=0 b=1 < tmp1.su > din.su  ",
  "           rm -f tmp1.su;                                           ",
  " # get rid of 75% of the traces                                     ",
  "           sudecimate < din.su > din_dec.su dec=0.75 verbose=1      ",
  " # reconstruct the data                                             ",
  "           sualft < din_dec.su > dout.su dh=1 niter=10 verbose=1 fmax=80 Ltw=250",
  " # plot results                                                     ",
  " suxwigb < din.su     key=offset clip=2 title='True data'      label2='Offset (m)'  label1='Time (s)'& ",
  " suxwigb < din_dec.su key=offset clip=2 title='Decimated data' label2='Offset (m)'  label1='Time (s)'& ",
  " suxwigb < dout.su    key=offset clip=2 title='ALFT result'    label2='Offset (m)'  label1='Time (s)'& ",
  "                                                                   ",
  " References:                                                       ",
  " Xu, S., Y. Zhang, D. Pham, and G. Lambaré, 2005b, Antileakage     ",
  " Fourier transform for seismic data regularization: Geophysics, 70,",
  " no.	4, V87–V95, doi: 10.1190/1.1993713.                           ",
  "                                                                   ",
  "  Future development:                                              ",
  "    allow for user to specify min/max of output coord              ",
  "    allow for user to choose coordinates to regularize (s,g,h,m)   ",
  "    allow for coordinate transformation if doing 2d reg along an azimuth",
  "    anti-alias capabilities                                        ",
  "    higher dimensions (3d,5d)                                      ",
  "                                                                   ",
 NULL};
/* Credits:
 * Aaron Stanton
 * Trace header fields accessed: dt, ns, offset
 * Last changes: May : 2013 
 */
/**************** end self doc ***********************************/

segy tr;
int main(int argc, char **argv)
{
  int verbose;
  time_t start,finish;
  double elapsed_time;
  int it,ix,nt,nx,nx_out,method;
  float dt,dh,hmin,hmax;
  float *h,*h_out;
  float **din,**dout,**din_tw,**dout_tw;
  int *ih,*ih_out;
  int padt,padx;
  int Ltw,Dtw;   
  int twstart;
  float taper;
  int itw,Itw,Ntw,niter;
  float fmin,fmax;

  /********/    
  fprintf(stderr,"*******SUALFT*********\n");
  /* Initialize */
  initargs(argc, argv);
  requestdoc(1);
  start=time(0);    
  /* Get parameters */
  if (!getparint("verbose", &verbose)) verbose = 0;
  if (!getparint("nx", &nx)) nx = 10000;
  if (!getparfloat("dh", &dh)) dh = 10;
  if (!gettr(&tr)) err("can't read first trace");
  if (!tr.dt) err("dt header field must be set");
  if (!tr.ns) err("ns header field must be set");
  if (!getparint("Ltw", &Ltw))  Ltw = 200; /* length of time window in samples */
  if (!getparint("Dtw", &Dtw))  Dtw = 10; /* overlap of time windows in samples	*/
  dt   = ((float) tr.dt)/1000000.0;
  nt = (int) tr.ns;
  if (!getparint("padt", &padt)) padt = 2; /* padding factor in time dimension*/
  if (!getparint("padx", &padx)) padx = 2; /* padding factor in spatial dimension*/
  if (!getparfloat("fmin",&fmin)) fmin = 0;
  if (!getparfloat("fmax",&fmax)) fmax = 0.5/dt;
  if (!getparint("niter", &niter)) niter = 100;
  fmax = MIN(fmax,0.5/dt);

  din   = ealloc2float(nt,nx);
  h        = ealloc1float(nx);
  ih       = ealloc1int(nx);
  /* ***********************************************************************
  input data
  *********************************************************************** */
  ix=0;
  do {
    h[ix]=(float)  tr.offset;
    memcpy((void *) din[ix],(const void *) tr.data,nt*sizeof(float));
    ix++;
    if (ix > nx) err("Number of traces > %d\n",nx); 
  } while (gettr(&tr));
  erewind(stdin);
  nx=ix;
  if (verbose) fprintf(stderr,"processing %d traces \n", nx);
  hmin = h[0];
  hmax = h[0];  
 
  for (ix=0;ix<nx;ix++){
  	if (hmin>h[ix]) hmin = h[ix]; 
  	if (hmax<h[ix]) hmax = h[ix]; 
  }
  for (ix=0;ix<nx;ix++){
  	ih[ix] = (int) trunc((h[ix]-hmin)/dh);
  }
  nx_out = 0;
  for (ix=0;ix<nx;ix++){
  	if (nx_out<ih[ix]) nx_out = ih[ix] + 1; 
  }
 
  ih_out = ealloc1int(nx_out);
  h_out = ealloc1float(nx_out);

  for (ix=0;ix<nx_out;ix++){
  	ih_out[ix] = ix;
  	h_out[ix] = ix*dh + hmin;
  }

  dout  = ealloc2float(nt,nx_out);

  Ntw = 9999;	
  /* number of time windows (will be updated during first 
  iteration to be consistent with total number of time samples
  and the length of each window) */
  
  din_tw = ealloc2float(Ltw,nx);
  dout_tw = ealloc2float(Ltw,nx_out);

/***********************************************************************
process using sliding time windows
***********************************************************************/
 twstart = 0;
 taper = 0;
 for (Itw=0;Itw<Ntw;Itw++){	
   if (Itw == 0){
	 Ntw = (int) trunc(nt/(Ltw-Dtw));
	 if ( (float) nt/(Ltw-Dtw) - (float) Ntw > 0) Ntw++;
   }		
   twstart = (int) Itw * (int) (Ltw-Dtw);
   if ((twstart+Ltw-1 >nt) && (Ntw > 1)){
   	 twstart=nt-Ltw;
   }
   if (Itw*(Ltw-Dtw+1) > nt){
      Ltw = (int) Ltw + nt - Itw*(Ltw-Dtw+1);
   }
   for (ix=0;ix<nx;ix++){ 
     for (itw=0;itw<Ltw;itw++){
       din_tw[ix][itw] = din[ix][twstart+itw];
     }
   }
   fprintf(stderr,"processing time window %d of %d\n",Itw+1,Ntw);
   process_time_window(din_tw,dout_tw,h,h_out,hmin,hmax,dt,Ltw,nx,nx_out,fmin,fmax,niter,padt,padx,verbose); 
   if (Itw==0){ 
     for (ix=0;ix<nx_out;ix++){ 
       for (itw=0;itw<Ltw;itw++){   
	     dout[ix][twstart+itw] = dout_tw[ix][itw];
       }	 	 
     }
   }
   else{
     for (ix=0;ix<nx_out;ix++){ 
       for (itw=0;itw<Dtw;itw++){   /* taper the top of the time window */
	     taper = (float) ((Dtw-1) - itw)/(Dtw-1); 
	     dout[ix][twstart+itw] = dout[ix][twstart+itw]*(taper) + dout_tw[ix][itw]*(1-taper);
       }
       for (itw=Dtw;itw<Ltw;itw++){   
	     dout[ix][twstart+itw] = dout_tw[ix][itw];
       }
     }	 	 
   }
 }
 /***********************************************************************
 end of processing time windows
 ***********************************************************************/

  /* ***********************************************************************
  output data
  *********************************************************************** */
  rewind(stdin);
  for (ix=0;ix<nx_out;ix++){ 
    memcpy((void *) tr.data,(const void *) dout[ix],nt*sizeof(float));
    tr.offset=(int) h_out[ix];
    tr.ntr=nx_out;
    tr.ns=nt;
    tr.dt = NINT(dt*1000000.);
    tr.tracl = ix+1;
    tr.tracr = ix+1;    
    fputtr(stdout,&tr);
  }
  
  /******** End of output **********/
  finish=time(0);
  elapsed_time=difftime(finish,start);
  fprintf(stderr,"Total time required: %6.2fs\n", elapsed_time);
  
  free1float(h);
  free1float(h_out);
  free2float(din);
  free2float(dout);
  free1int(ih);
  free1int(ih_out);
  free2float(din_tw);
  free2float(dout_tw);
  
  return EXIT_SUCCESS;
}

