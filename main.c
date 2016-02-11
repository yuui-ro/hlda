#include "utils.h"
#include "typedefs.h"
#include "doc.h"
#include "topic.h"
#include "gibbs.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define MAX_ITER 3500
#define TEST_LAG 100
#define NRESTARTS 1
#define TRAIN_NUM_SAMPLE 1
#define TRAIN_NUM_SPACE 100
#define TRAIN_BURNIN 3000
#define TEST_NUM_SAMPLE 10
#define TEST_BURNIN 3000


static inline double log_addition(double logx1, double logx2) {
  return logx1<logx2? log(exp(logx1-logx2)+1)+logx2 : log(1+exp(logx2-logx1))+logx1;
}


// simple gibbs sampling on a data set

void main_gibbs(int ac, char* av[])
{
    assert(ac == 5);

    char* corpus = av[2];
    char* settings = av[3];
    char* out_dir = av[4];

    int restart;
    for (restart = 0; restart < NRESTARTS; restart++)
    {
        gibbs_state* state =
            init_gibbs_state_w_rep(corpus, settings, out_dir, -1);
        int iter;
        for (iter = 0; iter < MAX_ITER; iter++)
        {
            iterate_gibbs_state(state, 1);
        }
        free_gibbs_state(state);
    }
}

void main_heldout(int ac, char* av[])
{
    assert(ac == 6);

    char* train = av[2];
    char* test = av[3];
    char* settings = av[4];
    char* out_dir = av[5];

    gibbs_state* state = init_gibbs_state_w_rep(train, settings, out_dir, -1);
    corpus* heldout_corp = corpus_new(state->corp->gem_mean,
                                      state->corp->gem_scale);
    read_corpus(test, heldout_corp, state->tr->depth);

    char filename[100];
    sprintf(filename, "%s/test.dat", state->run_dir);
    FILE* test_log = fopen(filename, "w");
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++)
    {
        iterate_gibbs_state(state, 1);
        if ((state->iter % TEST_LAG) == 0)
        {
            double score = mean_heldout_score(heldout_corp, state,
                                              200, 1, 1000);
            fprintf(test_log, "%04d %10.3f %d\n",
                    state->iter, score, ntopics_in_tree(state->tr));
            fflush(test_log);
        }
    }
    fclose(test_log);
}

static double eval_sample_harmonic_mean(double **testloglik,
                                        int num_sample,
                                        int num_test)
{
  double *sampleloglik = malloc(sizeof(double)*num_sample);
  double hm = 0.0;
  int i,j;
  for(i=0; i<num_sample; i++) {
    sampleloglik[i] = 0.0;
    for(j=0; j<num_test; j++) {
      sampleloglik[i] += testloglik[i][j];
    }
    hm += sampleloglik[i];
  }
  hm /= TRAIN_NUM_SAMPLE;

  free(sampleloglik);
  
  return hm;
}

void main_heldout_harmonic_mean(int ac, char *av[]) {
  assert(ac == 6);

  char* corpus_file = av[2];
  int train_size;
  sscanf(av[3], "%d", &train_size);
  char* settings = av[4];
  char* out_dir = av[5];

  gibbs_state* state = init_gibbs_state_w_rep(corpus_file, settings, out_dir, train_size);
  int test_size = state->corp->ndoc - train_size;
  
  // indexed by <sample_no, doc_no>
  double **testloglik = malloc(sizeof(double*)*TRAIN_NUM_SAMPLE); 
  int i, j;
  for(i=0; i<TRAIN_NUM_SAMPLE; i++) {
    testloglik[i] = malloc(sizeof(double)*test_size);
    for(j=0; j<test_size; j++) {
      testloglik[i][j] = -100;
    }
  }


  char filename[100];
  sprintf(filename, "%s/heldout_harmonic_mean.dat", state->run_dir);
  FILE* test_log = fopen(filename, "w");

  int iter;
  for (iter = 0; iter < TRAIN_BURNIN+TRAIN_NUM_SAMPLE*TRAIN_NUM_SPACE; iter++)
    {
      iterate_gibbs_state(state, 1);
      if ( iter >= TRAIN_BURNIN && (iter-TRAIN_BURNIN) % TRAIN_NUM_SPACE == 0 )
        {
          int postsample_no = (iter-TRAIN_BURNIN) / TRAIN_NUM_SPACE;
          // set the active data to frozen state
          set_frozen_state(state->corp);

          double before_loglik = state->eta_score;
          
          // evaluate each test doc
          for(i=0; i<test_size; i++)
            {
              // generate a new model state for each test document
              gibbs_state* test_state = new_heldout_gibbs_state(state->corp, state);
              
              // first set the doc to be evaluated to active
              doc *test_d = test_state->corp->doc[i+train_size];
              assert(test_d->state==heldout);
              test_d->state=active;

              // initialize the doc to be evaluated
              init_gibbs_state(test_state);

              // run burn-in
              for(j=0; j<TEST_BURNIN; j++)
                {
                  iterate_gibbs_state(test_state, 0);
                }
              
              // collect eta scores
              double after_loglik[TEST_NUM_SAMPLE];
              for(j=0; j<TEST_NUM_SAMPLE; j++)
                {
                  iterate_gibbs_state(test_state, 0);
                  after_loglik[j] = test_state->eta_score;
                  double ll = state->eta_score - before_loglik;
                  testloglik[postsample_no][i] = log_addition(testloglik[postsample_no][i], -ll);
                }
              testloglik[postsample_no][i] += log(TEST_NUM_SAMPLE);
              test_d->state=heldout;

              // free the generated test model state
              free_tree(test_state->tr);
              free(state);
            }          
        }
    }
  
  double hm;
  hm = eval_sample_harmonic_mean(testloglik, TRAIN_NUM_SAMPLE, test_size);
  fprintf(test_log, "%f", hm);
  fclose(test_log);
}

int main(int ac, char* av[])
{
    if (ac > 1)
    {
        if (strcmp(av[1], "gibbs") == 0)
        {
            main_gibbs(ac, av);
            return(0);
        }
        else if (strcmp(av[1], "heldout") == 0)
        {
            main_heldout(ac, av);
            return(0);
        }
    }
    outlog("USAGE: ./main gibbs corpus settings out");
    outlog("       ./main heldout train test settings out");
    return(0);
}
