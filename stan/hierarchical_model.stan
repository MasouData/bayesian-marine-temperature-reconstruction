data {
    int<lower=1> N;
    int<lower=1> J;

    vector[N] x;
    vector[N] temperature;

    array[N] int<lower=1, upper=J> species_id;
}

parameters {
    // Population-level parameters
    real a;
    real b;

    // Between-species variability
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;

    // Non-centered species effects
    vector[J] z_a;
    vector[J] z_b;

    // Residual uncertainty
    real<lower=0> sigma_T;
}

transformed parameters {
    vector[J] a_j;
    vector[J] b_j;

    a_j = a + sigma_a * z_a;
    b_j = b + sigma_b * z_b;
}

model {
    // Weakly informative population priors
    a ~ normal(15, 10);
    b ~ normal(0, 5);

    // Weakly informative priors on between-species variation
    sigma_a ~ normal(0, 5);
    sigma_b ~ normal(0, 2);

    // Non-centered parameterization
    z_a ~ std_normal();
    z_b ~ std_normal();

    sigma_T ~ exponential(0.5);

    // Likelihood
    for (n in 1:N) {
        temperature[n] ~ normal(
            a_j[species_id[n]]
            + b_j[species_id[n]] * x[n],
            sigma_T
        );
    }
}

generated quantities {
    vector[N] temperature_rep;
    vector[N] log_lik;

    for (n in 1:N) {
        temperature_rep[n] = normal_rng(
            a_j[species_id[n]]
            + b_j[species_id[n]] * x[n],
            sigma_T
        );

        log_lik[n] = normal_lpdf(
            temperature[n]
            | a_j[species_id[n]]
            + b_j[species_id[n]] * x[n],
            sigma_T
        );
    }
}