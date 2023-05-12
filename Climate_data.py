# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:19:26 2023

@author: Olayinka Abolade
"""
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err
from scipy.optimize import curve_fit


"""i am choosing GDP per capita (current US$)' and 'CO2 emissions (kt)' as the
 features to cluster on was made based on the goal of the analysis, which is 
 to find interesting clusters of data related to climate change"""

# Load World Bank data and select the relevant indicator for analysis
#('CO2 emissions (metric tons per capita)') for analysis 
Worldbank_data = pd.read_csv("climate_data.csv", 
                             skiprows=4)
co2_indicator = 'CO2 emissions (metric tons per capita)'
df_co2 = Worldbank_data[Worldbank_data['Indicator Name'] == co2_indicator]
print(df_co2)
# Load GDP per capita dataset
df_gdp = pd.read_csv('GDP_ per_ capita.csv', skiprows=4)
print(df_gdp)
print(df_co2.describe())
print(df_gdp.describe())

# drop rows with nan's in 2019
df_co2 = df_co2.dropna(subset=["2019"])
print(df_co2.describe)
df_gdp = df_gdp.dropna(subset=["2019"])
print(df_gdp.describe)

# Select relevant columns for analysis
df_co2_2019 = df_co2[["Country Name", "Country Code", "2019"]].copy()
df_gdp_2019 = df_gdp[["Country Name", "Country Code", "2019"]].copy()
print(df_co2_2019.describe())
print(df_gdp_2019.describe())

# Merge dataframes; df_co2_2019 and df_gdp_2019 dataframes
data_2019 = pd.merge(df_co2_2019, df_gdp_2019, on="Country Name", how="outer")
print(data_2019.describe())
data_2019.to_csv("CO2_GDP_2019.csv")

# Remove rows with missing values
data_2019 = data_2019.dropna() 

# rename columns
data_2019 = data_2019.rename(columns={"country": "Country Name", 
                                      "2019_x":"co2_per_capita", 
                                      "2019_y":"gdp_per_capita"})
print(data_2019.head())
print(data_2019.describe())
# Check for correlation 
pd.plotting.scatter_matrix(data_2019,figsize=(12, 12), s=5, alpha=0.8)
print(data_2019.corr())

# Extract columns for clustering
data_cluster = data_2019[["Country Name", "co2_per_capita", 
                          "gdp_per_capita"]].copy()
print(data_cluster.head())
print(data_cluster.describe())

# Normalize dataframe
data_cluster, df_min, df_max = ct.scaler(data_cluster[["co2_per_capita",
                                                       "gdp_per_capita"]])
# Determine optimal number of clusters using silhouette score
sil_scores = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data_cluster.iloc[:, 1:])
    score = skmet.silhouette_score(data_cluster.iloc[:, 1:], labels)
    sil_scores.append(score)
    print(f"Silhouette score for {n_clusters} clusters: {score:.3f}")

# Plot silhouette scores
plt.plot(range(2, 10), sil_scores, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.show()
# select  2 and 3 clusters based on the highest Silhouette score of 0.783 
# Perform clustering using KMeans with 3 clusters
nc = 3 # number of cluster centres
kmeans = cluster.KMeans(n_clusters=nc)
# Fit the data, results are stored in the kmeans object
kmeans.fit(data_cluster) # fit done on x,y pairs
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
# Visualize the clusters using a scatter plot
plt.figure(figsize=(8.0, 8.0))
plt.scatter(data_cluster["gdp_per_capita"], data_cluster["co2_per_capita"], 
            c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours
# show cluster centres
xc = cen[:,0]
yc = cen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
plt.xlabel("gdp_per_capita")
plt.ylabel("co2_per_capita")
plt.title("Scatter plot of CO2 and GDP per capita for 3 clusters")
plt.show()

# Plot the original data and cluster centers
nc = 3 # number of cluster centres
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(data_cluster)
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(8.0, 8.0))
# scatter plot with colours selected using the cluster numbers
# now using the original dataframe
plt.scatter(data_2019["gdp_per_capita"], data_2019["co2_per_capita"],
            c=labels, cmap="tab10")
# rescale and show cluster centres
scen = ct.backscale(cen, df_min, df_max)
xc = scen[:,0]
yc = scen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
plt.xlabel("gdp_per_capita")
plt.ylabel("co2_per_capita")
plt.title("Scatter plot of CO2 & GDP per capita(unnormalized data)for 3clusters")
plt.show()

# Perform clustering using KMeans with 2 clusters
nc = 2 # number of cluster centres
kmeans = cluster.KMeans(n_clusters=nc)
# Fit the data, results are stored in the kmeans object
kmeans.fit(data_cluster) # fit done on x,y pairs
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
# Visualize the clusters using a scatter plot
plt.figure(figsize=(8.0, 8.0))
plt.scatter(data_cluster["gdp_per_capita"], data_cluster["co2_per_capita"], 
            c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours
# show cluster centres
xc = cen[:,0]
yc = cen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
# c = colour, s = size
plt.xlabel("gdp_per_capita")
plt.ylabel("co2_per_capita")
plt.title("Scatter plot of CO2 and GDP per capita for 2 clusters")
plt.show()
# Add "Cluster" column to the main dataframe
data_cluster = data_2019[["Country Name", "co2_per_capita", 
                          "gdp_per_capita"]].copy()
data_cluster["Cluster"] = labels

# Define a function to print countries in each cluster
def print_countries_in_clusters(data):
    unique_clusters = data["Cluster"].unique()
    for cluster_id in unique_clusters:
        cluster_data = data[data["Cluster"] == cluster_id]
        countries = cluster_data["Country Name"].values
        print(f"Countries in Cluster {cluster_id}:")
        for country in countries:
            print(country)
        print()
print_countries_in_clusters(data_cluster)

#select two countries: one from cluster 0 and the other from cluster 1 
countries = ['Canada', 'Nigeria']
# Perform curve fitting for each country
# Define the logistic function
#select two countries: one from cluster 0 and the other from cluster 1 
## Country list for GDP per capital analysis
GDP = pd.read_csv("GDP_data.csv", skiprows=4)
GDP = GDP.dropna()
GDP.drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
GDP.set_index('Country Name', drop=True, inplace=True)
GDP
countries = ['Canada','Nigeria']
GDP_countries = GDP.loc[countries]
GDP_countries = GDP_countries.transpose()
GDP_countries = GDP_countries.rename_axis('Year')
GDP_countries = GDP_countries.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
GDP_countries = GDP_countries.dropna()
print(GDP_countries)

def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f

# Convert index to numeric and use it in curve fitting
GDP_countries.index = pd.to_numeric(GDP_countries.index)

popt, pcorr = opt.curve_fit(logistics, GDP_countries.index, 
                            GDP_countries["Canada"], p0=(16e8, 0.04, 1985.0))
print("Fit parameter", popt)

# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))

GDP_countries["GDP_logistics"] = logistics(GDP_countries.index, *popt)

# call function to calculate upper and lower limits with extrapolation
# create extended year range
years = np.arange(1960, 2040)
lower, upper = err.err_ranges(years, logistics, popt, sigmas)

plt.figure()
plt.title("logistics function")
plt.plot(GDP_countries.index, GDP_countries["Canada"], label="data")
plt.plot(GDP_countries.index, GDP_countries["GDP_logistics"], label="fit")

# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5)

plt.legend(loc="upper left")
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 10)) 

countries = ['Canada', 'Nigeria']
# Different initial parameters for each country
p0_values = [(16e8, 0.04, 1985.0), (16e8, 0.03, 1980.0)] 

# Apply the curve fitting for each country
for i, country in enumerate(countries):
    popt, pcorr = opt.curve_fit(logistics, GDP_countries.index, 
                                GDP_countries[country], p0=p0_values[i], 
                                maxfev=10000)

    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))

    # create extended year range
    years = np.arange(1960, 2040)

    # Calculate the fitted GDP and the confidence intervals
    GDP_logistics = logistics(years, *popt)
    lower, upper = err.err_ranges(years, logistics, popt, sigmas)

    # Plot the original data, the fitted function, and the confidence ranges
    axs[i].plot(GDP_countries.index, GDP_countries[country], label="data")
    axs[i].plot(years, GDP_logistics, label="fit")
    axs[i].fill_between(years, lower, upper, alpha=0.5)

    axs[i].set_title(f"logistics function for {country}")
    axs[i].legend(loc="upper left")

plt.tight_layout()  # adjust the subplots to fit in to the figure area.
plt.show()

def poly(t, c0, c1, c2, c3):
    """ Computes a polynomial c0 + c1*t + c2*t^2 + c3*t^3
    """
    t = t - 1950
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

# Set GDP_countries.index 
popt, pcorr = opt.curve_fit(poly, GDP_countries.index, GDP_countries["Canada"])
print("Fit parameter", popt)

# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))

# call function to calculate upper and lower limits with extrapolation
# create extended year range
years = np.arange(1960, 2040)
lower, upper = err.err_ranges(years, poly, popt, sigmas)

GDP_countries["poly"] = poly(GDP_countries.index, *popt)

plt.figure()
plt.title("polynomial")
plt.plot(GDP_countries.index, GDP_countries["Canada"], label="data")
plt.plot(GDP_countries.index, GDP_countries["poly"], label="fit")

# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5)

plt.legend(loc="upper left")
plt.show()
