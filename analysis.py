import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# Data cleaning
def clean_column_names(df):
    def standardize(col):
        # Add underscore between lowercase-uppercase (e.g., productURL → product_URL)
        col = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', col)
        # Lowercase, replace spaces with _, and remove non-alphanumerics except _
        col = col.strip().lower().replace(' ', '_')
        col = re.sub(r'[^\w_]', '', col)
        return col

    df.columns = [standardize(col) for col in df.columns]
    return df

def filter_publisher(df, publisher_name="Penguin"):
    filtered = df[df['sold_by'].str.contains(publisher_name, case=False, na=False)].copy()
    return filtered

def clean_and_preprocess(df):
    df['price'] = df['price'].astype(str).str.replace('$', '').astype(float)
    df['stars'] = pd.to_numeric(df['stars'], errors='coerce')
    df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce').fillna(0)
    df['title'] = df['title'].fillna('')

    df['price'] = df['price'].round(2)
    df['stars'] = df['stars'].round(2)
    return df

def check_reviews(df): # reviews seem to have a lot of 0 values -- checking sum to see if null
    total = df['reviews'].sum()
    print(f"Sum of all values in reviews: {total}")

# EDA
def extract_keywords(df, max_features=50):
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df['title'])
    keywords = tfidf.get_feature_names_out()
    keywords_df = pd.DataFrame(tfidf_matrix.toarray(), columns=keywords)
    return keywords_df, keywords

def plot_correlation_matrix(df, save_path="Results/correlation_matrix.png"):
    numeric_cols = ['price', 'stars', 'reviews']
    corr_df = df[numeric_cols].dropna()

    corr = corr_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix: Price, Stars, Reviews")
    plt.tight_layout()

    plt.savefig(save_path)

def plot_price_vs_rating(df, save_path="Results/price_vs_rating.png"):
    """ Price vs. Rating - Do higher-rated books cost more?"""
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='stars', y='price', alpha=0.3)
    sns.regplot(data=df, x='stars', y='price', scatter=False, color='red')
    plt.title("Price vs. Rating")
    plt.xlabel("Average Star Rating")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_price_vs_reviews(df, save_path="Results/price_vs_reviews.png"):
    """ Price vs. Reviews - Do more popular books cost more?"""
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='reviews', y='price', alpha=0.3)
    sns.regplot(data=df, x='reviews', y='price', scatter=False, color='orange')
    plt.xscale('log')
    plt.title("Price vs. Number of Reviews (log scale)")
    plt.xlabel("Number of Reviews (log)")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_price_by_year(df, save_path="Results/price_by_year.png"):
    """Price Over Time - Are newer books cheaper or more expensive?"""
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df['year'] = df['published_date'].dt.year
    yearly_avg = df.groupby('year')['price'].mean().reset_index()

    plt.figure(figsize=(10,6))
    sns.lineplot(data=yearly_avg, x='year', y='price')
    plt.title("Average Price by Year of Publication")
    plt.xlabel("Year")
    plt.ylabel("Average Price ($)")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_price_by_tags(df, save_path="Results/price_by_tags.png"):
    """Check if being a "bestseller"/“editor's pick” justifies a higher price"""
    melted = df.melt(
        value_vars=['is_best_seller', 'is_editors_pick', 'is_good_reads_choice'],
        var_name='Feature',
        value_name='Is_Flagged'
    )
    melted = melted.join(df[['price']])
    plt.figure(figsize=(10,6))
    sns.boxplot(data=melted[melted['Is_Flagged']], x='Feature', y='price')
    plt.title("Impact of Special Tags on Price")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.savefig(save_path)

# ML model
def build_pricing_model(df, keywords_df):
    features = pd.concat([df[['stars', 'reviews']].reset_index(drop=True), keywords_df], axis=1)
    target = df['price'].fillna(df['price'].median())
    features = features.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    mae = float(mae)
    
    return model, rmse, features.columns, y_test, y_pred, mae

# Plot model efficiency
def plot_feature_importance(model, feature_names, save_path="Results/features.png", top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = feature_names[indices][:top_n]
    top_importances = importances[indices][:top_n]

    plt.figure(figsize=(10,6))
    plt.title("Top 10 Features Impacting E-book Price")
    plt.barh(top_features[::-1], top_importances[::-1])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_distributions(df, save_dir="Results"):
    """ To understand the spread, skew, or shape of individual variables
      - how many books are priced around $10, or whether most ratings are clustered around 4.5."""
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title("Price Distribution of Penguin E-books")

    plt.subplot(1,2,2)
    sns.histplot(df['stars'], bins=30, kde=True)
    plt.title("Rating Distribution of Penguin E-books")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "price_rating_distribution.png"))

def plot_actual_vs_predicted(y_test, y_pred, save_path="Results/actual_vs_predicted.png"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
    sns.lineplot(x=y_test, y=y_test, color='red', label='Ideal Prediction')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs. Predicted Book Prices")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_error_distribution(y_test, y_pred, mae, save_path="Results/error_distribution.png"):
    errors = np.abs(y_test - y_pred)

    plt.figure(figsize=(8,5))
    sns.histplot(errors, bins=30, kde=True, color='skyblue')
    plt.axvline(mae, color='red', linestyle='--', label=f"MAE = ${mae:.2f}")
    plt.title("Distribution of Absolute Prediction Errors")
    plt.xlabel("Absolute Error ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

def plot_error_boxplot(y_test, y_pred, save_path="Results/error_boxplot.png"):
    errors = np.abs(y_test - y_pred)

    plt.figure(figsize=(6,4))
    sns.boxplot(errors, color='lightgreen')
    plt.title("Boxplot of Absolute Prediction Errors")
    plt.xlabel("Absolute Error ($)")
    plt.tight_layout()
    plt.savefig(save_path)

def plot_actual_vs_predicted_with_mae_band(y_test, y_pred, mae, save_path="Results/actual_vs_predicted_mae.png"):

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', label='Perfect Prediction')

    plt.fill_between(
        [y_test.min(), y_test.max()],
        [y_test.min() - mae, y_test.max() - mae],
        [y_test.min() + mae, y_test.max() + mae],
        color='red', alpha=0.2, label=f"±MAE (${mae:.2f})"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs. Predicted Prices with MAE Error Band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)

# Export data
def save_summary_to_file(df, keywords, rmse, output_path="Results/summary.txt"):
    summary = []

    summary.append(f"Number of Penguin books: {len(df)}\n")  # Add dataset size
 
    summary.append("Summary Statistics:\n")     # Add stats
    summary.append(df[['price', 'stars', 'reviews']].describe().to_string())
    summary.append("\n")

    summary.append("Top keywords in Penguin book titles:\n") # Add keywords
    summary.append(", ".join(keywords))
    summary.append("\n")

    summary.append(f"RMSE on test set: ${rmse:.2f}\n")  # Add RMSE
    summary.append(f"MAE on test set: ${mae:.2f}\n")  # Add MAE

    with open(output_path, "w") as f:
        f.write("\n".join(summary))

if __name__ == "__main__":
    os.makedirs('Results', exist_ok=True)

    df = pd.read_csv('raw_kindle_data.csv')
    df = clean_column_names(df)
    df.to_csv('penguin_kindle_data.csv', index=False)
    penguin_df = filter_publisher(df)

    penguin_df = clean_and_preprocess(penguin_df)
    check_reviews(penguin_df)

    keywords_df, keywords = extract_keywords(penguin_df)
    plot_correlation_matrix(penguin_df)
    plot_price_vs_rating(penguin_df)
    plot_price_vs_reviews(penguin_df)
    plot_price_by_year(penguin_df)
    plot_price_by_tags(penguin_df)

    model, rmse, feature_names, y_test, y_pred, mae = build_pricing_model(penguin_df, keywords_df)
    print(f"RMSE on test set: ${rmse:.2f}")
    print(f"MAE on test set: ${mae:.2f}")
    plot_actual_vs_predicted(y_test, y_pred)


    plot_feature_importance(model, feature_names)
    plot_distributions(penguin_df)

    save_summary_to_file(penguin_df, keywords, rmse)

    plot_error_distribution(y_test, y_pred, mae)
    plot_error_boxplot(y_test, y_pred)
    plot_actual_vs_predicted_with_mae_band(y_test, y_pred, mae)

