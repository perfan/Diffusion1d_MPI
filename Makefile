## Makefile for diffusion1d
##
## Ramses van Zon 2020
##

CXX=mpicxx
CXXFLAGS=-O3 -g -march=native -std=c++14
LDFLAGS=-g
LD=${CXX}

.PHONY: all clean distclean run 

all: diffusion1d

help: Makefile
	@sed -n 's/^##//p' $<

##  diffusion1d: build diffusion1d executable
diffusion1d: diffusion1d.o diffusion1d_timestep.o diffusion1d_output.o parameters.o ticktock.o sparkline.o
	${LD} ${LDFLAGS} -o diffusion1d diffusion1d.o diffusion1d_timestep.o diffusion1d_output.o parameters.o ticktock.o sparkline.o ${LDLIBS}

diffusion1d.o: diffusion1d.cc diffusion1d_timestep.h
	${CXX} ${CXXFLAGS} -c -o diffusion1d.o diffusion1d.cc

parameters.o: parameters.cc parameters.h
	${CXX} ${CXXFLAGS} -c -o parameters.o parameters.cc

diffusion1d_output.o: diffusion1d_output.cc diffusion1d_output.h
	${CXX} ${CXXFLAGS} -c -o diffusion1d_output.o diffusion1d_output.cc

diffusion1d_timestep.o: diffusion1d_timestep.cc diffusion1d_timestep.h
	${CXX} ${CXXFLAGS} -c -o diffusion1d_timestep.o diffusion1d_timestep.cc

ticktock.o: ticktock.cc ticktock.h sparkline.h
	${CXX} ${CXXFLAGS} -c -o ticktock.o ticktock.cc

sparkline.o: sparkline.cc sparkline.h sparkline.h
	${CXX} ${CXXFLAGS} -c -o sparkline.o sparkline.cc

##  clean: Remove auto-generated files
clean:
	\rm -f ticktock.o diffusion1d.o parameters.o diffusion1d_output.o diffusion1d_timestep.o parameters.o sparkline.o

##  distclean: Remove auto-generated files, executables, and documentation
distclean: clean
	\rm -rf output.dat diffusion1d Doxyfile* html latex

##  run: execute diffusion1d with parameters in params.ini
run: diffusion1d params.ini
	./diffusion1d params.ini

##  doc: use doxygen to generate documentation in latex/refman.pdf
doc:
	doxygen -g
	sed -i 's/PROJECT_NAME[ ]*=.*/PROJECT_NAME=Diffusion1D/' Doxyfile
	doxygen
	make -C latex
