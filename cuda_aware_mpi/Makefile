M_TARGETS := all_to_all


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