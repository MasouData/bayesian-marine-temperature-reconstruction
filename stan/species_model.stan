data {
    int<lower=1> N;
    int<lower=1> J;

    vector[N] x;
    vector[N] temperature;

    array[N] int<lower=1, upper=J> species_id;
}

parameters {
    vector[J] a;
    vector[J] b;

    real<lower=0> sigma_T;
}

model {
    // Weakly informative species-level priors
    a ~ normal(15, 10);
    b ~ normal(0, 5);

    sigma_T ~ exponential(0.5);

    // Species-specific likelihood
    for (n in 1:N) {
        temperature[n] ~ normal(
            a[species_id[n]]
            + b[species_id[n]] * x[n],
            sigma_T
        );
    }
}

generated quantities {
    vector[N] temperature_rep;
    vector[N] log_lik;

    for (n in 1:N) {

        temperature_rep[n] = normal_rng(
            a[species_id[n]]
            + b[species_id[n]] * x[n],
            sigma_T
        );

        log_lik[n] = normal_lpdf(
            temperature[n]
            | a[species_id[n]]
            + b[species_id[n]] * x[n],
            sigma_T
        );
    }
}