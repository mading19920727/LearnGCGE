
CC = gcc
LIBBLASLAPACK = -L/usr/lib -lopenblas -lm
OBJ      = main.o test_app_ccs.o  test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o ../src/ops_multi_grid.o ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_ccs.o ../app/app_lapack.o 
LINKOBJ  = main.o test_app_ccs.o  test_eig_sol_gcg.o ../src/ops.o ../src/ops_eig_sol_gcg.o ../src/ops_lin_sol.o ../src/ops_multi_grid.o ../src/ops_multi_vec.o ../src/ops_orth.o ../app/app_ccs.o ../app/app_lapack.o 
LIBS     = $(LIBBLASLAPACK)
INCS     = -I../src -I../app -I/usr/include
CXXINCS  = 
BIN      = TestOPS.exe
CXXFLAGS = $(CXXINCS) -O2 -Wall -fPIC -g
CFLAGS   = $(INCS) -O2 -Wall -fPIC -g -fopenmp #-fopenmp -DMEMWATCH -DMW_STDIO ##
RM       = rm -f

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	${RM} $(OBJ) $(BIN) 

$(BIN): $(OBJ)
	$(CC) $(LINKOBJ) -o $(BIN) $(LIBS)

main.o: main.c
	$(CC) -c main.c -o main.o $(CFLAGS)

test_app_ccs.o: test_app_ccs.c
	$(CC) -c test_app_ccs.c -o test_app_ccs.o $(CFLAGS)

test_eig_sol_gcg.o: test_eig_sol_gcg.c
	$(CC) -c test_eig_sol_gcg.c -o test_eig_sol_gcg.o $(CFLAGS)

../src/ops.o: ../src/ops.c
	$(CC) -c ../src/ops.c -o ../src/ops.o $(CFLAGS)

../src/ops_eig_sol_gcg.o: ../src/ops_eig_sol_gcg.c
	$(CC) -c ../src/ops_eig_sol_gcg.c -o ../src/ops_eig_sol_gcg.o $(CFLAGS)

../src/ops_lin_sol.o: ../src/ops_lin_sol.c
	$(CC) -c ../src/ops_lin_sol.c -o ../src/ops_lin_sol.o $(CFLAGS)

../src/ops_multi_grid.o: ../src/ops_multi_grid.c
	$(CC) -c ../src/ops_multi_grid.c -o ../src/ops_multi_grid.o $(CFLAGS)

../src/ops_multi_vec.o: ../src/ops_multi_vec.c
	$(CC) -c ../src/ops_multi_vec.c -o ../src/ops_multi_vec.o $(CFLAGS)

../src/ops_orth.o: ../src/ops_orth.c
	$(CC) -c ../src/ops_orth.c -o ../src/ops_orth.o $(CFLAGS)

../app/app_ccs.o: ../app/app_ccs.c
	$(CC) -c ../app/app_ccs.c -o ../app/app_ccs.o $(CFLAGS)

../app/app_lapack.o: ../app/app_lapack.c
	$(CC) -c ../app/app_lapack.c -o ../app/app_lapack.o $(CFLAGS)
