#!/bin/sh

suwaveform type=ricker1 fpeak=30 | sugain pbal=1 > wavelet.su

suplane nt=200 npl=2 ntr=500 dip1=0 dip2=1 len1=500 len2=500 | suconv sufile=wavelet.su > tmp1.su

suchw key1=offset key2=tracl a=0 b=1 < tmp1.su > tmp2.su

mv tmp2.su din.su

rm -f tmp*.su wavelet.su

