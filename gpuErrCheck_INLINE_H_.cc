
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#define TRACE_MAX_LENGTH 10
void mhzGetBackTrace()
{
    void *array[TRACE_MAX_LENGTH];
    size_t size = backtrace(array,TRACE_MAX_LENGTH);
    char **strings = backtrace_symbols(array, size);

    fprintf(stderr,"Back Trace:\n");
    for (int i = 0; i < size; i++)
	fprintf(stderr, "%s\n",strings[i]);
}
