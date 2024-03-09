M_TARGETS := single_thread_all_to_all single_thread_all_reduce mpi_all_to_all mpi_all_reduce


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