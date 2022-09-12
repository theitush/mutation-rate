from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")
import numpy as np
from sbi import utils as sbiutils
from sbi.inference.base import infer
import pickle
import multiprocessing as mp
from params import readable_prior

def simulate(params, syn_prob=0.31, passages=10, pop_size=10**9, default_sample_size=2000, genome_length=3560, 
             seq_error_rate=0.00005, return_data=False, sample_sizes=None):
    # imports within the function are required by dill for muliprocessing
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from scipy.stats import poisson
    import itertools
    import time
    from scipy.special import factorial
    import torch
    
    def multinomial_sampling(freqs_dict, sample_size):
        freqs_after_sample = np.random.multinomial(sample_size, list(freqs_dict.values()))/sample_size
        freqs_dict = {key: val for key, val in zip(freqs_dict.keys(), freqs_after_sample) if val>0}
        return freqs_dict

    def get_poisson_probs(mutation_rate, min_freq):
        probs = dict()
        for num_of_muts in range(10000):  # just a large number coz while loops are slow and awkward
            prob = poisson.pmf(num_of_muts, mutation_rate)
            if prob <= min_freq:
                break
            probs[num_of_muts] = prob
        return probs


    def generate_possible_GCs(mut_num):
        GCs_unformatted = itertools.combinations_with_replacement(['syn', 'non_syn', 'syn_ada', 'non_syn_ada'], 
                                                                  mut_num)
        possible_GCs = list()
        for muts_combo in GCs_unformatted:
            possible_GCs.append((muts_combo.count('syn'), muts_combo.count('non_syn'),
                                 muts_combo.count('syn_ada'), muts_combo.count('non_syn_ada')))
        return possible_GCs


    def calc_mutations_probs(mut_num, poisson_prob, ps, min_freq):
        GCs_array = generate_possible_GCs(mut_num)
        probabilities = np.product(np.power(ps, GCs_array), axis=1) 
        factorials = factorial(mut_num)/np.product(factorial(GCs_array), axis=1)
        multinomial_prob = factorials * probabilities * poisson_prob
        return {GC: prob for GC, prob in zip(GCs_array, multinomial_prob) if prob>min_freq}


    def get_mutations(mutation_rate, syn_ratio, p_ada_syn, p_ada_non_syn, min_freq):
        ps = [syn_ratio - p_ada_syn, 
              1 - syn_ratio - p_ada_non_syn, 
              p_ada_syn, 
              p_ada_non_syn]
        mut_poisson_prob = get_poisson_probs(mutation_rate, min_freq)
        mutations = dict()
        for mut_num, poisson_prob in mut_poisson_prob.items():
            mutations.update(calc_mutations_probs(mut_num, poisson_prob, ps, min_freq))
        return mutations


    def simulate_p0(p0_syn, p0_non_syn, pop_size, max_ada_per_genome):
        p0_sum_of_mutations = p0_syn + p0_non_syn
        p0_syn_ratio = p0_syn / p0_sum_of_mutations
        # we assume no adaptive mutations at p0
        p0_mutations = get_mutations(p0_sum_of_mutations, p0_syn_ratio, 0, 0, 1/(100*pop_size))
        # convert to 6-tuple 
        p0_mutations = {mut+(0,0) :freq for mut, freq in p0_mutations.items()}
        return p0_mutations

    def get_epistatsis(muts_by_fitness, epistasis_boost):
        # give penalty value to genotypes with more than one adaptive mutation

        multiple_adaptive_idx = np.argwhere(muts_by_fitness[:,4]>1).reshape(-1)
        multiple_adaptive = muts_by_fitness[multiple_adaptive_idx,4]
        fitness_len = muts_by_fitness.shape[0]
        epistasis = [epistasis_boost if x in multiple_adaptive_idx else 1 for x in range(fitness_len)]
        return epistasis

    def selection(fitness_effects, muts_by_fitness, freqs, epistasis_boost):
        start = time.time()
        no_epi_fitness = np.product(np.power(fitness_effects, muts_by_fitness), axis=1).reshape(-1)
        epi_effects = fitness_effects.copy()
        epi_effects[4] =  epi_effects[4] ** epistasis_boost
        epistasis_fitness = np.product(np.power(epi_effects, muts_by_fitness), axis=1).reshape(-1)
        fitness = np.where(muts_by_fitness[:,4]>1, epistasis_fitness, no_epi_fitness)
        avg_fitness = np.sum(freqs*fitness)
        fitness /= avg_fitness
        return fitness

    def gather_muts_by_fitness(genotypes):
        primordial = genotypes[:,:2]
        no_adas = genotypes[:,2:4]
        just_adas = np.sum(genotypes[:,4:], axis=1).reshape(-1,1)
        return np.concatenate([primordial, no_adas, just_adas], axis=1)


    def mutate_and_select(genotypes, genotypes_freqs, mutations, mutations_freqs, fitness_effects, 
                          max_ada_per_genome, tuple_size, epistasis_boost):
        # do that numpy magic:
        new_genotypes = genotypes + mutations
        new_genotypes = new_genotypes.reshape(-1,tuple_size)
        new_freqs = genotypes_freqs * mutations_freqs                          # mutation
        new_freqs = new_freqs.reshape(-1)
        muts_by_fitness = gather_muts_by_fitness(new_genotypes)
        fitness = selection(fitness_effects, muts_by_fitness, new_freqs, epistasis_boost) 
        new_genotypes = list(map(tuple, new_genotypes))
        new_freqs = new_freqs * fitness
        return new_genotypes, new_freqs


    def simulate_next_passage(fitness_effects, passage, mutations, pop_size, max_ada_per_genome, epistasis_boost):
        # turn dict into arrays:
        tuple_size = len(list(passage.keys())[0])
        genotypes = np.array(list(passage.keys()), dtype=int).reshape(-1,1,tuple_size)
        genotypes_freqs = np.array(list(passage.values()), dtype=float).reshape(-1,1,1)
        new_genotypes, new_freqs = mutate_and_select(genotypes, genotypes_freqs, np.array(list(mutations.keys())), 
                                                     np.array(list(mutations.values())), fitness_effects, 
                                                     max_ada_per_genome, tuple_size, epistasis_boost)
        freqs_dict = defaultdict(float)
        for mut, freq in zip(new_genotypes, new_freqs):
            freqs_dict[mut] += freq
        freqs_dict = {key: val for key, val in freqs_dict.items() if val > 1/(pop_size*1000)}  # to prevent occasional bugs
        freqs_sum = sum(freqs_dict.values())
        freqs_dict = {key: val/freqs_sum for key, val in freqs_dict.items()}
        freqs_dict = multinomial_sampling(freqs_dict, pop_size)               # drift
        return freqs_dict

    def get_short_sumstat(df, passages=[3,7,10]):
        ret = list()
        for passage in passages:
            ret.append(sum(df['syn_total'] * df[passage]))
            ret.append(sum(df['non_syn_total'] * df[passage]))
        return torch.Tensor(ret)

    def get_manual_stats(df, passages=[3,7,10]):
        max_muts = 11 # so 10 maximum muts
        new_index = [(x,y,z,w) for w in range(max_muts) for z in range(max_muts)
                     for y in range(max_muts) for x in range(max_muts) if x+y<max_muts and z<=x and w<=y]
        grouped = df.groupby(['syn_total', 'non_syn_total', 'syn_ben', 'non_syn_ben'])[passages].sum()
        return torch.Tensor(grouped.reindex(new_index).fillna(0).values.flatten())

    def get_med_stats(df, passages=[3,7,10]):
        max_muts = 11 # so 10 maximum muts
        new_index = [(x,y) for y in range(max_muts) for x in range(max_muts) if x+y<max_muts]
        grouped = df.groupby(['syn_total', 'non_syn_total'])[passages].sum()
        return torch.Tensor(grouped.reindex(new_index).fillna(0).values.flatten())

    def get_total_sumstat(df, passages=[3,7,10]):
        return torch.cat((get_short_sumstat(df, passages), get_med_stats(df, passages), get_manual_stats(df, passages)))

    def wrangle_data(passage):
        data = pd.DataFrame(passage)
        data['mut_num'] = [sum(x) for x in data.index]
        data = data.reset_index().rename(columns={'level_5': 'non_syn_ben', 'level_1': 'non_syn_pri',
                                                  'level_4': 'syn_ben', 'level_0': 'syn_pri',
                                                  'level_3': 'non_syn', 'level_2': 'syn'}).fillna(0)
        data['syn_non_ben'] = data['syn'] + data['syn_pri']
        data['non_syn_non_ben'] = data['non_syn'] + data['non_syn_pri']
        data['syn_total'] = data['syn_non_ben'] + data['syn_ben']
        data['non_syn_total'] = data['non_syn_non_ben'] + data['non_syn_ben']
        return data
   
    def simulate_sequence_sampling(passages, sample_sizes, seq_error_rate, syn_prob, p_ada_syn, p_ada_non_syn, 
                               max_ada_per_genome, default_sample_size):
        sequenced_passages = dict()
        if sample_sizes is None:
            sample_size = default_sample_size
            seq_errors = get_mutations(sample_size*seq_error_rate, syn_prob, p_ada_syn, p_ada_non_syn, 
                                       1/(100*sample_size))
            seq_errors = {(0,0)+mut :freq for mut, freq in seq_errors.items()}
        fitness_effects = np.ones(5)
        for i in range(len(passages)):
            if sample_sizes is not None:
                if i==3:
                    sample_size = sample_sizes[0]
                elif i==7:
                    sample_size = sample_sizes[1]
                elif i==10:
                    sample_size = sample_sizes[2]
                else:
                    sample_size = default_sample_size
                seq_errors = get_mutations(sample_size*seq_error_rate, syn_prob, p_ada_syn, p_ada_non_syn, 
                                   1/(100*sample_size))
                seq_errors = {(0,0)+mut :freq for mut, freq in seq_errors.items()}
            sequenced_passages[i] = simulate_next_passage(fitness_effects, passage[i], seq_errors, sample_size, 
                                               max_ada_per_genome, 0)
        return sequenced_passages
    
    try:
        start = time.time()
        mutation_rate = 10 ** params[0]
        w_syn = params[1]
        w_non_syn = params[2]
        w_ada = params[3]
        p_ada_syn = params[4]
        p_ada_non_syn = params[5]
        p0_syn = params[6]
        p0_non_syn = params[7]
        w_penalty = params[8]
        epistasis_boost = params[9]
        
        fitness_effects = np.array([w_syn**w_penalty, w_non_syn**w_penalty, w_syn, w_non_syn, w_ada])
        max_ada_per_genome = {'reg': p_ada_non_syn*genome_length, 
                              'syn': p_ada_syn*genome_length}
        passage = dict()
        passage[0] = simulate_p0(p0_syn, p0_non_syn, pop_size, max_ada_per_genome)
        mutations = get_mutations(mutation_rate, syn_prob, p_ada_syn, p_ada_non_syn, 1/(100*pop_size))
        mutations = {(0,0)+mut :freq for mut, freq in mutations.items()}
        for i in range(passages):
            passage[i+1] = simulate_next_passage(fitness_effects, passage[i], mutations, pop_size, max_ada_per_genome,
                                                 epistasis_boost)
        sequenced_passages = simulate_sequence_sampling(passage, sample_sizes, seq_error_rate, syn_prob, p_ada_syn, p_ada_non_syn, 
                               max_ada_per_genome, default_sample_size)
        data = wrangle_data(sequenced_passages)
        if not return_data:
            data = get_total_sumstat(data) 
    
    except Exception as e:
        raise Exception(f"Exception: '{e}' occured with params: {params}")
    
    return data