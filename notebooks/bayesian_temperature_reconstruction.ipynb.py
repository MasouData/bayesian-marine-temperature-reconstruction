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