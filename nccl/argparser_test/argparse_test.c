#include "../util/argparse.h"

static struct options opts;
static struct parser_doc parser_doc;

int main(int argc, char *argv[]) {

  default_parser_doc("Single thread all reduce to test nccl working", "1",
                     &parser_doc);
  argument_parse(&opts, &parser_doc, argc, argv);

  printf("OPTS: \n"
         "DATA_LEN %d\n"
         "DATA_TYPE: %d\n"
         "NUM_GPU: %d\n"
         "ITER: %d\n"
         "WARMUP: %d\n",
         opts.data_len, opts.data_type, opts.num_gpus, opts.iterations,
         opts.warmup_iterations);

  return 0;
}