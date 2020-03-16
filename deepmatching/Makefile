CC=g++

OS_NAME=$(shell uname -s)
ifeq ($(OS_NAME),Linux) 

#create libsatlas.so
#su -
#cd /usr/lib/x86_64-linux-gnu/
#ld -shared -o libsatlas.so --whole-archive libatlas.a liblapack.a --no-whole-archive libf77blas.a libcblas.a
  LAPACKLDFLAGS=/usr/lib/x86_64-linux-gnu/libsatlas.so  # single-threaded blas
#install libgfortran and create /usr/lib/x86_64-linux-gnu/libgfortran.so
  FORTRANFLAG=/usr/lib/x86_64-linux-gnu/libgfortran.so
  
endif
ifeq ($(OS_NAME),Darwin)  # Mac OS X
  LAPACKLDFLAGS=-framework Accelerate # for OS X
endif
LAPACKCFLAGS=-Dinteger=int $(BLAS_THREADING)
#make sure you have correct path for .a files
STATICLAPACKLDFLAGS=-fPIC -Wall -g -fopenmp -static -static-libstdc++ /usr/lib/x86_64-linux-gnu/libjpeg.a /usr/lib/x86_64-linux-gnu/libpng.a /usr/lib/x86_64-linux-gnu/libz.a /usr/lib/x86_64-linux-gnu/libblas.a /usr/lib/gcc/x86_64-linux-gnu/7/libgfortran.a /usr/lib/gcc/x86_64-linux-gnu/7/libquadmath.a # statically linked version

CFLAGS= -fPIC -Wall -g -std=c++11 $(LAPACKCFLAGS) -fopenmp -DUSE_OPENMP -O3
LDFLAGS=-fPIC -Wall -g -ljpeg -lpng -fopenmp
CPYTHONFLAGS=-I/usr/include/python3.6
LIBFLAGS=-lm -ljpeg -lpng

SOURCES := $(shell find . -name '*.cpp' ! -name 'deepmatching_matlab.cpp')
OBJ := $(SOURCES:%.cpp=%.o)
HEADERS := $(shell find . -name '*.h')


all: deepmatching

.cpp.o:  %.cpp %.h
	$(CC) -o $@ $(CFLAGS) -c $+

deepmatching: $(HEADERS) $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(FORTRANFLAG) #fortran AFTER atlas !!

deepmatching-static: $(HEADERS) $(OBJ)
	$(CC) -o $@ $^ $(STATICLAPACKLDFLAGS)

python: $(HEADERS) $(OBJ)
#doesn't work !!!
#	swig -python $(CPYTHONFLAGS) deepmatching.i # not necessary, only do if you have swig compiler
	$(CC) $(CFLAGS) -c deepmatching_wrap.c $(CPYTHONFLAGS)
	$(CC) -shared $(LDFLAGS) $(LAPACKLDFLAGS) deepmatching_wrap.o $(OBJ) -o _deepmatching.so $(LIBFLAGS) $(LAPACKLDFLAGS) $(FORTRANFLAG) 

clean:
	rm -f $(OBJ) deepmatching *~ *.pyc .gdb_history deepmatching_wrap.o _deepmatching.so deepmatching.mex???
