#include "argparse.h"
#include "simple_utils.h"
#include <argp.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

error_t parse(int key, char *arg, struct argp_state *state) {
  struct options *arg_options = state->input;
  switch (key) {
  case 'n': // num gpus
    arg_options->num_gpus = atoi(arg);
    break;
  case 'i': // num iters
    arg_options->iterations = atoi(arg);
    break;
  case 'w':
    arg_options->warmup_iterations = atoi(arg);
    break;
  case 'd':
    arg_options->data_len = atoi(arg);
    break;
  case 't': // data type
    if (strcmp(arg, "FLOAT") == 0) {
      arg_options->data_type = OPTION_FLOAT;
    } else if (strcmp(arg, "INT") == 0) {
      arg_options->data_type = OPTION_INT;
    } else if (strcmp(arg, "CHAR") == 0) {
      arg_options->data_type = OPTION_CHAR;
    } else {
      return ARGP_KEY_ERROR;
    }
    break;
  case ARGP_KEY_ARG:
    return 0;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

void report_options(struct options *opt) {
  REPORT("\n****************\n"
         "Options:\n"
         "DATA LENGHT: %d\n"
         "DATA TYPE: %d\n"
         "NUM_GPUS: %d\n"
         "ITERATIONS: %d\n"
         "WARMUP ITERATIONS: %d\n"
         "****************\n",
         opt->data_len, opt->data_type, opt->num_gpus, opt->iterations,
         opt->warmup_iterations);
}

void argument_parse(struct options *options_struct,
                    struct parser_doc *parser_doc, int argc, char **argv) {

  default_options(options_struct);
  struct argp _argp = {options, parse, parser_doc->args_doc, parser_doc->doc, 0,
                       0,       0};
  memcpy(&argp, &_argp, sizeof(argp));
  argp_parse(&argp, argc, argv, 0, 0, options_struct);
}

void default_options(struct options *options) {
  options->data_len = 128;
  options->data_type = OPTION_FLOAT;
  options->iterations = 20;
  options->warmup_iterations = 0;
}

void build_parser_doc(char *doc, char *args_doc, char *version, char *email,
                      struct parser_doc *parser) {
  memset(parser, 0, sizeof(struct parser_doc));
  strncpy(parser->doc, doc, sizeof(parser->doc) - 1);
  strncpy(parser->args_doc, args_doc, sizeof(parser->args_doc) - 1);
  strncpy(parser->version, version, sizeof(parser->version) - 1);
  strncpy(parser->email, email, sizeof(parser->email) - 1);
}

void default_parser_doc(char *doc, char *version, struct parser_doc *parser) {
  build_parser_doc(doc, "-n NUM_GPUS", version, "egencer20@ku.edu.tr", parser);
}
