#include <stdio.h>
#include <stdlib.h>
#include <math.h>
extern "C"{
#include "ops.h"
}

extern "C" int TestAppCCS   (int argc, char *argv[]);

int main(int argc, char *argv[])
{
	TestAppCCS(argc, argv);
   	return 0;
}