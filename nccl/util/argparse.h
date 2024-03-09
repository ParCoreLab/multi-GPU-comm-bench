#include <argp.h>

#define MAX_DOC_LEN 256

struct parser_doc {
  char doc[MAX_DOC_LEN];
  char args_doc[MAX_DOC_LEN];
  char version[64];
  char email[64];
};

struct options {
  unsigned int data_len;
  enum { OPTION_FLOAT, OPTION_INT, OPTION_CHAR } data_type;
  unsigned int num_gpus;
  unsigned int iterations;
  unsigned int warmup_iterations;
};

void report_options(struct options *opt);

error_t parse(int key, char *arg, struct argp_state *state);
void argument_parse(struct options *options_struct,
                    struct parser_doc *parser_doc, int argc, char *argv[]);
void default_options(struct options *options);

static struct argp_option options[] = {
    {"num-gpus", 'n', "N", 0, "Number of GPUS (ignored if MPI process)", 0},
    {"num-iter", 'i', "N", 0,
     "Number of iterations in the main iteration loop.", 0},
    {"num-warmup-iter", 'w', "N", 0,
     "Number of iterations in the warmup section. 0 by default.", 0},
    {"data-len", 'd', "N", 0, "Lenght of the data block.", 0},
    {"data-type", 't', "FLOAT|INT|CHAR", 0, "Type of the data block.", 0},
    {0}};

static struct argp argp;

void build_parser_doc(char *doc, char *args_doc, char *version, char *email,
                      struct parser_doc *parser);

void default_parser_doc(char *doc, char *version, struct parser_doc *parser);