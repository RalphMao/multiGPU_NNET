#ifndef gpuErrCheck_INLINE_H_

extern void mhzGetBackTrace();
#define gpuErrCheck(ans) { gpuAssert( (ans), __FILE__, __LINE__ ); }
inline void gpuAssert( cudaError_t code, const char *file, int line, bool abort = true) {

   if (code != cudaSuccess) {
      fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      mhzGetBackTrace();
      if (abort) 
      	exit(code);
   }
}


#endif
