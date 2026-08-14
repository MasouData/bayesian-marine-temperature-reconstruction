data {
    int<lower=1> N;
    vector[N] x;
    vector[N] temperature;
}

parameters {
    real a;
    real b;
    real<lower=0> sigma_T;
}

model {
    // Weakly informative priors
    a ~ normal(15, 10);
    b ~ normal(0, 5);
    sigma_T ~ exponential(0.5);

    // Likelihood
    temperature ~ normal(a + b * x, sigma_T);
}

generated quantities {
    vector[N] temperature_rep;
    vector[N] log_lik;

    for (n in 1:N) {
        temperature_rep[n] =
            normal_rng(a + b * x[n], sigma_T);

        log_lik[n] =
            normal_lpdf(
                temperature[n]
                | a + b * x[n],
                sigma_T
            );
    }
}