#include "utils.h"
#include "typedefs.h"
#include "doc.h"
#include "topic.h"
#include "gibbs.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define MAX_ITER 30
#define TEST_LAG 30
#define NRESTARTS 1

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
                init_gibbs_state_w_rep(corpus, settings, out_dir);
        int iter;
        for (iter = 0; iter < MAX_ITER; iter++)
        {
            iterate_gibbs_state(state);
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

    gibbs_state* state = init_gibbs_state_w_rep(train, settings, out_dir);
    corpus* heldout_corp = corpus_new(state->corp->gem_mean,
                                      state->corp->gem_scale);
    read_corpus(test, heldout_corp, state->tr->depth);

    char filename[100];
    sprintf(filename, "%s/test.dat", state->run_dir);
    FILE* test_log = fopen(filename, "w");
    sprintf(filename, "%s/test_perplexity.dat", state->run_dir);
    FILE* perplexity_log = fopen(filename, "w");
    sprintf(filename, "%s/test_numwords.dat", state->run_dir);
    FILE* numwords_log = fopen(filename, "w");
    sprintf(filename, "%s/test_loglik.dat", state->run_dir);
    FILE* loglik_log = fopen(filename, "w");

    int iter;
    for (iter = 0; iter < MAX_ITER; iter++)
    {
        iterate_gibbs_state(state);
        if ((state->iter % TEST_LAG) == 0)
        {
          double score, eta_score;
          int numwords;
            mean_heldout_score(heldout_corp, state,
                               300, 1, 200, &score, &eta_score);
            numwords = total_number_words(heldout_corp);
            fprintf(test_log, "%04d %10.3f %d\n",
                    state->iter, score, ntopics_in_tree(state->tr));
            fprintf(loglik_log, "%f", eta_score);
            fprintf(numwords_log, "%d", numwords);
            fprintf(perplexity_log, "%f", exp(-eta_score/numwords));
            fflush(test_log);
            fflush(loglik_log);
            fflush(numwords_log);
            fflush(perplexity_log);
        }
    }
    fclose(test_log);
    fclose(loglik_log);
    fclose(numwords_log);
    fclose(perplexity_log);
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
