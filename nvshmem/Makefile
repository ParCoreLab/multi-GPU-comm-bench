M_TARGETS := mpi_all_to_all mpi_all_to_all_builtin mpi_broadcast mpi_broadcast_kernel mpi_max_reduce mpi_sum_all_reduce


all: clean build

build:
	@for folder in $(M_TARGETS); do \
		echo "Making $$folder"; \
		(cd $$folder && $(MAKE)); \
	done
clean:
	@for folder in $(M_TARGETS); do \
		echo "Making $$folder"; \
		(cd $$folder && $(MAKE) clean); \
	done