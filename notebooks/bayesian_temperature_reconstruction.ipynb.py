# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Bayesian Marine Temperature Reconstruction
# MAGIC
# MAGIC ## Hierarchical Bayesian modelling of oxygen-isotope measurements
# MAGIC
# MAGIC This project investigates how marine carbonate formation temperature can
# MAGIC be reconstructed from oxygen-isotope measurements.
# MAGIC
# MAGIC The analysis compares three Bayesian modelling strategies:
# MAGIC
# MAGIC 1. Complete pooling — one temperature–isotope relationship for all species.
# MAGIC 2. No pooling — independent regression parameters for each species.
# MAGIC 3. Hierarchical partial pooling — species-specific relationships that share
# MAGIC    information through population-level distributions.
# MAGIC
# MAGIC The final hierarchical model is used for uncertainty-aware temperature
# MAGIC reconstruction.
# MAGIC
# MAGIC ### Research objective
# MAGIC
# MAGIC Model the relationship
# MAGIC
# MAGIC $$T_i = a + b(\delta^{18}O_{c,i}-\delta^{18}O_{w,i}) + \epsilon_i$$
# MAGIC
# MAGIC while investigating whether the intercept and slope vary systematically
# MAGIC between marine species.
# MAGIC
# MAGIC ### Reproducibility note
# MAGIC
# MAGIC The original dataset was supplied for Utrecht University coursework and is
# MAGIC therefore not distributed in this public repository.
# MAGIC
# MAGIC The repository contains the modelling code, Stan programs, validation logic,
# MAGIC figures, and documentation required to reproduce the analytical workflow with
# MAGIC a compatible dataset.

# COMMAND ----------

# MAGIC %pip install -q cmdstanpy

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cmdstanpy import CmdStanModel, cmdstan_path

SEED = 42
np.random.seed(SEED)

print("Environment ready.")

# COMMAND ----------

# DBTITLE 1,Install CmdStan binaries
import cmdstanpy

print("Installing CmdStan binaries...")
cmdstanpy.install_cmdstan(verbose=True)

print(f"\n✓ CmdStan installed at: {cmdstanpy.cmdstan_path()}")

# COMMAND ----------

# DBTITLE 1,Configuration
DATA_PATH = "/Volumes/cmdstanpy/cmdstanpy/cmdstanpy/merged_data.csv"

REQUIRED_COLUMNS = [
    "ID",
    "paper",
    "location",
    "species",
    "temperature",
    "d18_O_w",
    "d18_O",
    "d18_O_sd",
    "d18_O_w_sd",
    "class",
    "functional_group",
    "composition",
]

MODELLING_COLUMNS = [
    "species",
    "temperature",
    "d18_O_w",
    "d18_O",
    "d18_O_sd",
    "d18_O_w_sd",
]

print(f"Dataset: {DATA_PATH}")

# COMMAND ----------

# DBTITLE 1,Load the dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset was not found at:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

display(df.head())

# COMMAND ----------

# DBTITLE 1,data validation
# 1. Required schema
missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# 2. Missing values in modelling variables
missing_values = df[MODELLING_COLUMNS].isna().sum()

if missing_values.sum() > 0:
    raise ValueError(
        "Missing values detected in modelling columns:\n"
        + str(missing_values[missing_values > 0])
    )

# 3. Numeric validation
numeric_columns = [
    "temperature",
    "d18_O_w",
    "d18_O",
    "d18_O_sd",
    "d18_O_w_sd",
]

for col in numeric_columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"{col} must be numeric.")

# 4. Standard deviations cannot be negative
if (df["d18_O_sd"] < 0).any():
    raise ValueError("Negative d18_O measurement uncertainty detected.")

if (df["d18_O_w_sd"] < 0).any():
    raise ValueError("Negative d18_O_w measurement uncertainty detected.")

# 5. Species information is required
if df["species"].nunique() < 2:
    raise ValueError(
        "Hierarchical modelling requires observations from multiple species."
    )

print("✓ Schema validation passed")
print("✓ No missing modelling values")
print("✓ Numeric columns validated")
print("✓ Measurement uncertainties validated")
print("✓ Species structure validated")

# COMMAND ----------

# DBTITLE 1,Feature construction
df["isotope_diff"] = df["d18_O"] - df["d18_O_w"]

print(
    "Created isotope_diff = d18_O - d18_O_w"
)

display(
    df[
        [
            "species",
            "temperature",
            "d18_O",
            "d18_O_w",
            "isotope_diff",
        ]
    ].head()
)

# COMMAND ----------

# DBTITLE 1,Dataset summary
summary = pd.DataFrame({
    "Metric": [
        "Observations",
        "Species",
        "Locations",
        "Source papers",
        "Minimum temperature (°C)",
        "Maximum temperature (°C)",
    ],
    "Value": [
        len(df),
        df["species"].nunique(),
        df["location"].nunique(),
        df["paper"].nunique(),
        df["temperature"].min(),
        df["temperature"].max(),
    ],
})

display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory analysis
# MAGIC
# MAGIC Before fitting Bayesian models, I inspect the relationship between
# MAGIC oxygen-isotope difference and observed formation temperature.
# MAGIC
# MAGIC A key modelling question is whether a single temperature–isotope
# MAGIC relationship is adequate for all species, or whether species-specific
# MAGIC relationships are required.
# MAGIC
# MAGIC The exploratory analysis therefore focuses on:
# MAGIC
# MAGIC - representation of each species in the dataset;
# MAGIC - the range of observed temperatures;
# MAGIC - the relationship between isotope difference and temperature;
# MAGIC - visible heterogeneity between species.

# COMMAND ----------

# DBTITLE 1,Species-level exploratory summary
species_summary = (
    df.groupby("species")
    .agg(
        observations=("species", "size"),
        mean_temperature=("temperature", "mean"),
        min_temperature=("temperature", "min"),
        max_temperature=("temperature", "max"),
        mean_isotope_diff=("isotope_diff", "mean"),
    )
    .sort_values("observations", ascending=False)
    .reset_index()
)

species_summary[
    [
        "mean_temperature",
        "min_temperature",
        "max_temperature",
        "mean_isotope_diff",
    ]
] = species_summary[
    [
        "mean_temperature",
        "min_temperature",
        "max_temperature",
        "mean_isotope_diff",
    ]
].round(2)

display(species_summary)

# COMMAND ----------

# DBTITLE 1,Temperature–isotope relationship by species
# ============================================================
# Temperature–isotope relationship by species
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))

for species, group in df.groupby("species"):
    ax.scatter(
        group["isotope_diff"],
        group["temperature"],
        label=species,
        alpha=0.72,
        s=38,
    )

ax.set_xlabel(
    r"Isotope difference  $\delta^{18}O_c - \delta^{18}O_w$"
)
ax.set_ylabel("Formation temperature (°C)")

ax.set_title(
    "Marine Temperature vs Oxygen-Isotope Difference by Species"
)

ax.grid(alpha=0.2)

ax.legend(
    title="Species",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8,
)

fig.tight_layout()

plt.show()

# COMMAND ----------

FIGURES_DIR = os.path.abspath("../figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

figure_path = os.path.join(
    FIGURES_DIR,
    "isotope_temperature_by_species.png"
)

fig.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight",
)

print(f"Figure saved to:\n{figure_path}")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bayesian model 1 — Complete pooling
# MAGIC
# MAGIC We first establish a baseline model that assumes all observations,
# MAGIC regardless of species, follow the same temperature–isotope relationship.
# MAGIC
# MAGIC $$ T_i = a + b(\delta^{18}O_{c,i}-\delta^{18}O_{w,i})+\epsilon_i,$$
# MAGIC
# MAGIC where
# MAGIC
# MAGIC $$\epsilon_i \sim \mathcal{N}(0,\sigma_T).$$
# MAGIC
# MAGIC The complete-pooling model ignores species-specific heterogeneity.
# MAGIC It therefore provides a useful baseline against which the
# MAGIC species-specific and hierarchical models can be evaluated.
# MAGIC
# MAGIC Weakly informative priors are used for the regression parameters.
# MAGIC Posterior predictive samples and pointwise log-likelihood values are
# MAGIC generated for subsequent model checking and comparison.

# COMMAND ----------

# ============================================================
# prepare Stan data
# ============================================================

complete_pooling_data = {
    "N": len(df),
    "x": df["isotope_diff"].to_numpy(),
    "temperature": df["temperature"].to_numpy(),
}

print(f"N = {complete_pooling_data['N']}")
print(
    f"Isotope range: "
    f"{complete_pooling_data['x'].min():.2f} to "
    f"{complete_pooling_data['x'].max():.2f}"
)

print(
    f"Temperature range: "
    f"{complete_pooling_data['temperature'].min():.2f} to "
    f"{complete_pooling_data['temperature'].max():.2f} °C"
)

# COMMAND ----------

# ============================================================
# Compile complete-pooling Stan model
# ============================================================

COMPLETE_POOLING_STAN = os.path.abspath(
    "../stan/complete_pooling.stan"
)

print(f"Stan model:\n{COMPLETE_POOLING_STAN}")

complete_pooling_model = CmdStanModel(
    stan_file=COMPLETE_POOLING_STAN
)

print("✓ Complete-pooling model compiled successfully")

# COMMAND ----------

# DBTITLE 1,Fit complete-pooling model

complete_pooling_fit = complete_pooling_model.sample(
    data=complete_pooling_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=SEED,
    show_progress=True,
)

print("✓ Sampling completed")

# COMMAND ----------

print(complete_pooling_fit.diagnose())

# COMMAND ----------

# DBTITLE 1,Posterior parameter summary
# ============================================================
# Posterior parameter summary
# ============================================================

summary_cp = complete_pooling_fit.summary()

parameters_cp = summary_cp.loc[
    ["a", "b", "sigma_T"],
    [
        "Mean",
        "StdDev",
        "5%",
        "50%",
        "95%",
        "ESS_bulk",
        "R_hat",
    ],
]

display(parameters_cp.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Posterior predictive assessment
# MAGIC
# MAGIC Successful MCMC convergence does not by itself imply that a model adequately
# MAGIC represents the scientific structure of the data.
# MAGIC
# MAGIC We therefore evaluate the complete-pooling model using posterior predictive
# MAGIC checks. Replicated temperature datasets generated from the fitted model are
# MAGIC compared with the observed measurements.
# MAGIC
# MAGIC We additionally examine model residuals by species. Systematic species-level
# MAGIC residual patterns would indicate that a single common regression relationship
# MAGIC does not adequately capture species heterogeneity.

# COMMAND ----------

# ============================================================
# Posterior predictive check
# ============================================================

temperature_rep = complete_pooling_fit.stan_variable(
    "temperature_rep"
)

observed_temperature = df["temperature"].to_numpy()

# Statistics for observed data
observed_stats = {
    "Mean": np.mean(observed_temperature),
    "Std. deviation": np.std(observed_temperature),
    "Minimum": np.min(observed_temperature),
    "Maximum": np.max(observed_temperature),
}

# Same statistics for every posterior predictive dataset
replicated_stats = {
    "Mean": np.mean(temperature_rep, axis=1),
    "Std. deviation": np.std(temperature_rep, axis=1),
    "Minimum": np.min(temperature_rep, axis=1),
    "Maximum": np.max(temperature_rep, axis=1),
}

ppc_rows = []

for statistic, observed_value in observed_stats.items():
    simulated = replicated_stats[statistic]

    ppc_rows.append({
        "Statistic": statistic,
        "Observed": observed_value,
        "Predictive 5%": np.quantile(simulated, 0.05),
        "Predictive median": np.quantile(simulated, 0.50),
        "Predictive 95%": np.quantile(simulated, 0.95),
    })

ppc_summary = pd.DataFrame(ppc_rows).round(2)

display(ppc_summary)

# COMMAND ----------

# ============================================================
# Complete-pooling diagnostic visualization
# ============================================================

a_draws = complete_pooling_fit.stan_variable("a")
b_draws = complete_pooling_fit.stan_variable("b")
sigma_draws = complete_pooling_fit.stan_variable("sigma_T")

# ------------------------------------------------------------
# Posterior predictive regression curve
# ------------------------------------------------------------

x_grid = np.linspace(
    df["isotope_diff"].min(),
    df["isotope_diff"].max(),
    200
)

mu_draws = (
    a_draws[:, None]
    + b_draws[:, None] * x_grid[None, :]
)

rng = np.random.default_rng(SEED)

predictive_draws = rng.normal(
    loc=mu_draws,
    scale=sigma_draws[:, None]
)

mean_prediction = np.mean(mu_draws, axis=0)

predictive_lower = np.quantile(
    predictive_draws, 0.05, axis=0
)

predictive_upper = np.quantile(
    predictive_draws, 0.95, axis=0
)

# ------------------------------------------------------------
# Species-level residuals
# ------------------------------------------------------------

a_mean = np.mean(a_draws)
b_mean = np.mean(b_draws)

df["complete_pooling_prediction"] = (
    a_mean
    + b_mean * df["isotope_diff"]
)

df["complete_pooling_residual"] = (
    df["temperature"]
    - df["complete_pooling_prediction"]
)

species_residuals = (
    df.groupby("species")["complete_pooling_residual"]
    .mean()
    .sort_values()
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6)
)

# ---- Panel A: complete pooling fit ----

for species, group in df.groupby("species"):
    axes[0].scatter(
        group["isotope_diff"],
        group["temperature"],
        label=species,
        alpha=0.60,
        s=30,
    )

axes[0].plot(
    x_grid,
    mean_prediction,
    linewidth=2.5,
    label="Complete-pooling mean"
)

axes[0].fill_between(
    x_grid,
    predictive_lower,
    predictive_upper,
    alpha=0.18,
    label="90% posterior predictive interval"
)

axes[0].set_xlabel(
    r"Isotope difference  $\delta^{18}O_c-\delta^{18}O_w$"
)

axes[0].set_ylabel(
    "Formation temperature (°C)"
)

axes[0].set_title(
    "A. Complete-Pooling Bayesian Regression"
)

axes[0].grid(alpha=0.2)

# ---- Panel B: residuals by species ----

axes[1].barh(
    species_residuals.index,
    species_residuals.values
)

axes[1].axvline(
    0,
    linewidth=1.5
)

axes[1].set_xlabel(
    "Mean residual (observed − predicted temperature, °C)"
)

axes[1].set_title(
    "B. Systematic Residuals by Species"
)

axes[1].grid(
    axis="x",
    alpha=0.2
)

fig.suptitle(
    "Why Complete Pooling May Be Insufficient",
    fontsize=15
)

fig.tight_layout()

# Save README-quality figure
FIGURES_DIR = os.path.abspath("../figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

figure_path = os.path.join(
    FIGURES_DIR,
    "complete_pooling_diagnostics.png"
)

fig.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight"
)

print(
    f"Figure saved to:\n{figure_path}"
)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bayesian model 2 — Species-specific regression (no pooling)
# MAGIC
# MAGIC The complete-pooling model captures the overall temperature–isotope
# MAGIC relationship but exhibits systematic residual differences between species.
# MAGIC
# MAGIC We therefore allow each species \(j\) to have its own intercept and slope:
# MAGIC
# MAGIC $$T_i =a_{j[i]} + b_{j[i]}(\delta^{18}O_{c,i}-\delta^{18}O_{w,i})+\epsilon_i.$$
# MAGIC
# MAGIC This represents a **no-pooling model**: each species-specific relationship
# MAGIC is estimated independently, without sharing information between species.
# MAGIC
# MAGIC The model provides an important comparison with both the complete-pooling
# MAGIC baseline and the hierarchical partial-pooling model introduced later.

# COMMAND ----------

# ============================================================
# Species encoding for Stan
# ============================================================

species_names = sorted(df["species"].unique())

species_to_id = {
    species: i + 1
    for i, species in enumerate(species_names)
}

df["species_id"] = (
    df["species"]
    .map(species_to_id)
    .astype(int)
)

J = len(species_names)

species_mapping = pd.DataFrame({
    "species_id": range(1, J + 1),
    "species": species_names
})

display(species_mapping)

print(f"Number of species: {J}")

# COMMAND ----------

# ============================================================
# No-pooling Stan data
# ============================================================

species_model_data = {
    "N": len(df),
    "J": J,
    "x": df["isotope_diff"].to_numpy(),
    "temperature": df["temperature"].to_numpy(),
    "species_id": df["species_id"].to_numpy(),
}

print(f"N = {species_model_data['N']}")
print(f"J = {species_model_data['J']}")

print(
    "Species IDs:",
    np.unique(species_model_data["species_id"])
)

# COMMAND ----------

# ============================================================
# Compile species-specific model
# ============================================================

SPECIES_MODEL_STAN = os.path.abspath(
    "../stan/species_model.stan"
)

species_model = CmdStanModel(
    stan_file=SPECIES_MODEL_STAN
)

print("✓ Species-specific model compiled successfully")

# COMMAND ----------

# ============================================================
# Fit species-specific model
# ============================================================

species_fit = species_model.sample(
    data=species_model_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=SEED,
    show_progress=True,
)

print("✓ Sampling completed")

# COMMAND ----------

print(species_fit.diagnose())

# COMMAND ----------

# ============================================================
# Species-specific posterior estimates
# ============================================================

a_draws_species = species_fit.stan_variable("a")
b_draws_species = species_fit.stan_variable("b")

species_parameter_summary = pd.DataFrame({
    "Species": species_names,

    "Intercept mean": np.mean(
        a_draws_species,
        axis=0
    ),

    "Intercept SD": np.std(
        a_draws_species,
        axis=0
    ),

    "Slope mean": np.mean(
        b_draws_species,
        axis=0
    ),

    "Slope SD": np.std(
        b_draws_species,
        axis=0
    ),
})

species_parameter_summary = (
    species_parameter_summary.round(3)
)

display(species_parameter_summary)

# COMMAND ----------

sigma_species = species_fit.stan_variable(
    "sigma_T"
)

print(
    f"Complete-pooling sigma_T: "
    f"{complete_pooling_fit.stan_variable('sigma_T').mean():.3f}"
)

print(
    f"No-pooling sigma_T: "
    f"{sigma_species.mean():.3f}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bayesian model 3 — Hierarchical partial pooling
# MAGIC
# MAGIC The no-pooling model substantially reduces residual uncertainty, indicating
# MAGIC that the temperature–isotope relationship varies between species.
# MAGIC
# MAGIC However, species are represented by very different numbers of observations.
# MAGIC Estimating every species independently can therefore produce relatively
# MAGIC uncertain parameter estimates for sparsely sampled species.
# MAGIC
# MAGIC A hierarchical model provides a compromise between complete pooling and
# MAGIC no pooling.
# MAGIC
# MAGIC Species-specific intercepts and slopes are modelled as:
# MAGIC
# MAGIC $$a_j \sim \mathcal{N}(a,\sigma_a)$$
# MAGIC
# MAGIC $$b_j \sim \mathcal{N}(b,\sigma_b)$$
# MAGIC
# MAGIC where  a and b describe the population-level relationship and
# MAGIC $$\sigma_a$$ and $$\sigma_b$$ quantify between-species heterogeneity.
# MAGIC
# MAGIC This allows well-supported species to retain distinctive relationships while
# MAGIC more uncertain species can borrow statistical information from the overall
# MAGIC population.
# MAGIC
# MAGIC A non-centered parameterization is used for efficient Bayesian sampling.

# COMMAND ----------

# ============================================================
# Hierarchical partial-pooling data
# ============================================================

hierarchical_data = {
    "N": len(df),
    "J": J,
    "x": df["isotope_diff"].to_numpy(),
    "temperature": df["temperature"].to_numpy(),
    "species_id": df["species_id"].to_numpy(),
}

print(f"N = {hierarchical_data['N']}")
print(f"J = {hierarchical_data['J']}")

# COMMAND ----------

# ============================================================
# Compile hierarchical model
# ============================================================

HIERARCHICAL_STAN = os.path.abspath(
    "../stan/hierarchical_model.stan"
)

hierarchical_model = CmdStanModel(
    stan_file=HIERARCHICAL_STAN
)

print("✓ Hierarchical model compiled successfully")

# COMMAND ----------

# ============================================================
# Fit hierarchical model
# ============================================================

hierarchical_fit = hierarchical_model.sample(
    data=hierarchical_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=SEED,
    show_progress=True,
)

print("✓ Sampling completed")

# COMMAND ----------

print(hierarchical_fit.diagnose())

# COMMAND ----------

# ============================================================
# Hierarchical population-level parameters
# ============================================================

summary_h = hierarchical_fit.summary()

population_summary = summary_h.loc[
    [
        "a",
        "b",
        "sigma_a",
        "sigma_b",
        "sigma_T",
    ],
    [
        "Mean",
        "StdDev",
        "5%",
        "50%",
        "95%",
        "ESS_bulk",
        "R_hat",
    ],
]

display(population_summary.round(4))

# COMMAND ----------

# ============================================================
# Hierarchical species-specific estimates
# ============================================================

a_j_draws = hierarchical_fit.stan_variable("a_j")
b_j_draws = hierarchical_fit.stan_variable("b_j")

hierarchical_species_summary = pd.DataFrame({
    "Species": species_names,

    "Intercept mean": np.mean(
        a_j_draws,
        axis=0
    ),

    "Intercept SD": np.std(
        a_j_draws,
        axis=0
    ),

    "Slope mean": np.mean(
        b_j_draws,
        axis=0
    ),

    "Slope SD": np.std(
        b_j_draws,
        axis=0
    ),
})

hierarchical_species_summary = (
    hierarchical_species_summary.round(3)
)

display(hierarchical_species_summary)

# COMMAND ----------

sigma_hierarchical = (
    hierarchical_fit
    .stan_variable("sigma_T")
    .mean()
)

print(
    f"Complete pooling : "
    f"{complete_pooling_fit.stan_variable('sigma_T').mean():.3f} °C"
)

print(
    f"No pooling       : "
    f"{species_fit.stan_variable('sigma_T').mean():.3f} °C"
)

print(
    f"Partial pooling  : "
    f"{sigma_hierarchical:.3f} °C"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Partial pooling and species-level shrinkage
# MAGIC
# MAGIC The hierarchical model achieves almost the same residual fit as the
# MAGIC fully species-specific model while sharing information across species.
# MAGIC
# MAGIC Partial pooling has the greatest influence on species whose independent
# MAGIC parameter estimates are uncertain. Well-supported species remain close
# MAGIC to their no-pooling estimates, while uncertain estimates are pulled
# MAGIC toward the population-level relationship.
# MAGIC
# MAGIC The following figure compares the species-specific intercepts and slopes
# MAGIC estimated by the no-pooling and hierarchical models.

# COMMAND ----------

# ============================================================
# Visualising hierarchical shrinkage
# ============================================================

# Sample sizes
species_counts = (
    df.groupby("species")
    .size()
    .reindex(species_names)
)

# No-pooling posterior summaries
no_pool_a_mean = np.mean(a_draws_species, axis=0)
no_pool_a_sd = np.std(a_draws_species, axis=0)

no_pool_b_mean = np.mean(b_draws_species, axis=0)
no_pool_b_sd = np.std(b_draws_species, axis=0)

# Partial-pooling posterior summaries
partial_a_mean = np.mean(a_j_draws, axis=0)
partial_a_sd = np.std(a_j_draws, axis=0)

partial_b_mean = np.mean(b_j_draws, axis=0)
partial_b_sd = np.std(b_j_draws, axis=0)

# Population-level means
population_a = hierarchical_fit.stan_variable("a").mean()
population_b = hierarchical_fit.stan_variable("b").mean()

# Order species by sample size
order = np.argsort(species_counts.values)

ordered_species = np.array(species_names)[order]
ordered_counts = species_counts.values[order]

y = np.arange(J)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 7),
    sharey=True
)

# ==========================
# A. Intercepts
# ==========================

axes[0].errorbar(
    no_pool_a_mean[order],
    y + 0.12,
    xerr=no_pool_a_sd[order],
    fmt="o",
    capsize=3,
    label="No pooling"
)

axes[0].errorbar(
    partial_a_mean[order],
    y - 0.12,
    xerr=partial_a_sd[order],
    fmt="o",
    capsize=3,
    label="Partial pooling"
)

# Connect estimates to show shrinkage
for i, idx in enumerate(order):
    axes[0].plot(
        [
            no_pool_a_mean[idx],
            partial_a_mean[idx]
        ],
        [
            y[i] + 0.12,
            y[i] - 0.12
        ],
        alpha=0.35
    )

axes[0].axvline(
    population_a,
    linestyle="--",
    linewidth=1.5,
    label="Population mean"
)

axes[0].set_xlabel("Species intercept $a_j$ (°C)")
axes[0].set_title("A. Intercept shrinkage")
axes[0].grid(axis="x", alpha=0.2)

# ==========================
# B. Slopes
# ==========================

axes[1].errorbar(
    no_pool_b_mean[order],
    y + 0.12,
    xerr=no_pool_b_sd[order],
    fmt="o",
    capsize=3,
    label="No pooling"
)

axes[1].errorbar(
    partial_b_mean[order],
    y - 0.12,
    xerr=partial_b_sd[order],
    fmt="o",
    capsize=3,
    label="Partial pooling"
)

for i, idx in enumerate(order):
    axes[1].plot(
        [
            no_pool_b_mean[idx],
            partial_b_mean[idx]
        ],
        [
            y[i] + 0.12,
            y[i] - 0.12
        ],
        alpha=0.35
    )

axes[1].axvline(
    population_b,
    linestyle="--",
    linewidth=1.5,
    label="Population mean"
)

axes[1].set_xlabel(
    r"Species slope $b_j$ (°C per unit $\Delta^{18}O$)"
)

axes[1].set_title("B. Slope shrinkage")
axes[1].grid(axis="x", alpha=0.2)

# ==========================
# Species labels + N
# ==========================

species_labels = [
    f"{species}  (n={n})"
    for species, n
    in zip(ordered_species, ordered_counts)
]

axes[0].set_yticks(y)
axes[0].set_yticklabels(species_labels)

axes[0].legend()
axes[1].legend()

fig.suptitle(
    "Hierarchical Partial Pooling Stabilizes Species-Level Estimates",
    fontsize=15
)

fig.tight_layout()

# Save
FIGURES_DIR = os.path.abspath("../figures")

figure_path = os.path.join(
    FIGURES_DIR,
    "partial_pooling_shrinkage.png"
)

fig.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight"
)

print(f"Figure saved to:\n{figure_path}")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predictive model comparison
# MAGIC
# MAGIC The three modelling strategies are compared using Pareto-smoothed
# MAGIC importance-sampling leave-one-out cross-validation (PSIS-LOO).
# MAGIC
# MAGIC PSIS-LOO estimates expected out-of-sample predictive accuracy by
# MAGIC approximating the effect of leaving each observation out of model fitting.
# MAGIC
# MAGIC We compare:
# MAGIC
# MAGIC 1. Complete pooling
# MAGIC 2. Species-specific no pooling
# MAGIC 3. Hierarchical partial pooling
# MAGIC
# MAGIC Higher expected log predictive density (ELPD) indicates better predictive
# MAGIC performance.
# MAGIC
# MAGIC The Pareto-\(k\) diagnostics are also inspected to ensure that the
# MAGIC PSIS approximation is reliable.

# COMMAND ----------

# MAGIC %pip install -q arviz

# COMMAND ----------

import arviz as az

print(f"ArviZ version: {az.__version__}")

# COMMAND ----------

# ============================================================
# Convert CmdStanPy results to ArviZ InferenceData
# ============================================================

observed_data = {
    "temperature": df["temperature"].to_numpy()
}

idata_complete = az.from_cmdstanpy(
    posterior=complete_pooling_fit,
    posterior_predictive="temperature_rep",
    observed_data=observed_data,
    log_likelihood={
        "temperature": "log_lik"
    }
)

idata_no_pooling = az.from_cmdstanpy(
    posterior=species_fit,
    posterior_predictive="temperature_rep",
    observed_data=observed_data,
    log_likelihood={
        "temperature": "log_lik"
    }
)

idata_partial_pooling = az.from_cmdstanpy(
    posterior=hierarchical_fit,
    posterior_predictive="temperature_rep",
    observed_data=observed_data,
    log_likelihood={
        "temperature": "log_lik"
    }
)

print("✓ All three models converted to ArviZ InferenceData")

# COMMAND ----------

# ============================================================
# PSIS-LOO
# ============================================================

loo_complete = az.loo(
    idata_complete,
    var_name="temperature",
    pointwise=True
)

loo_no_pooling = az.loo(
    idata_no_pooling,
    var_name="temperature",
    pointwise=True
)

loo_partial_pooling = az.loo(
    idata_partial_pooling,
    var_name="temperature",
    pointwise=True
)

print("COMPLETE POOLING")
print(loo_complete)

print("\nNO POOLING")
print(loo_no_pooling)

print("\nPARTIAL POOLING")
print(loo_partial_pooling)

# COMMAND ----------

# ============================================================
# Predictive model comparison
# ============================================================

model_comparison = az.compare(
    {
        "Complete pooling": loo_complete,
        "No pooling": loo_no_pooling,
        "Partial pooling": loo_partial_pooling,
    }
)

display(model_comparison.round(2))

# COMMAND ----------

# ============================================================
# Inspect influential PSIS-LOO observations
# ============================================================

k_no_pool = np.asarray(
    loo_no_pooling.pareto_k
).ravel()

k_partial = np.asarray(
    loo_partial_pooling.pareto_k
).ravel()

idx_no = int(np.argmax(k_no_pool))
idx_partial = int(np.argmax(k_partial))

print(
    f"No pooling:     row={idx_no}, "
    f"max Pareto k={k_no_pool[idx_no]:.3f}"
)

print(
    f"Partial pooling: row={idx_partial}, "
    f"max Pareto k={k_partial[idx_partial]:.3f}"
)

influential_indices = sorted(
    set(
        np.where(k_no_pool > 0.70)[0].tolist()
        +
        np.where(k_partial > 0.70)[0].tolist()
    )
)

influential_observations = df.iloc[
    influential_indices
][
    [
        "species",
        "location",
        "temperature",
        "d18_O",
        "d18_O_w",
        "isotope_diff",
    ]
].copy()

influential_observations["pareto_k_no_pooling"] = [
    k_no_pool[i] for i in influential_indices
]

influential_observations["pareto_k_partial_pooling"] = [
    k_partial[i] for i in influential_indices
]

display(
    influential_observations.round(3)
)

# COMMAND ----------

model_summary = pd.DataFrame({
    "Model": [
        "Complete pooling",
        "No pooling",
        "Partial pooling",
    ],
    "ELPD-LOO": [
        loo_complete.elpd,
        loo_no_pooling.elpd,
        loo_partial_pooling.elpd,
    ],
    "SE": [
        loo_complete.se,
        loo_no_pooling.se,
        loo_partial_pooling.se,
    ],
    "p_LOO": [
        loo_complete.p,
        loo_no_pooling.p,
        loo_partial_pooling.p,
    ],
    "Max Pareto k": [
        np.max(np.asarray(loo_complete.pareto_k)),
        np.max(k_no_pool),
        np.max(k_partial),
    ],
})

display(
    model_summary.round(2)
)