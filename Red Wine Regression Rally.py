import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset
df = pd.read_csv('PS_dataset.csv')

## 1
# Count the number of wines for each quality score
quality_counts = df['quality'].value_counts().sort_index()

# Create the bar plot
plt.figure(figsize=(10, 6))
bars=quality_counts.plot(kind='bar', color='teal', edgecolor='black')

# Adding title and labels
plt.title('Count of Wines for Each Quality Score',fontweight='semibold' ,fontsize=20,color='orange')
plt.xlabel('Quality Score', fontsize=14)
plt.ylabel('Count of Wines', fontsize=14)
plt.xticks(rotation=0)  # Rotate x-axis labels if needed

# Adding frequency on top of each bar
for bar in bars.patches:
    plt.text(bar.get_x() + bar.get_width() / 2, 
             bar.get_height(), 
             int(bar.get_height()), 
             ha='center', 
             va='bottom', 
          fontsize=12)
# Show the plot
plt.tight_layout()
plt.show()

##2
plt.scatter(df['alcohol'], df['pH'], alpha=0.3, color='indigo')

plt.title('Scatter Plot of Alcohol vs. pH',fontweight='semibold', fontsize=24,color='red')
plt.xlabel('Alcohol Content (%)', fontsize=12)
plt.ylabel('pH Level', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)

correlation = np.corrcoef(df['alcohol'], df['pH'])[0, 1]
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

##3

# Create the histogram
n, bins, patches = plt.hist(df['alcohol'], 
                          bins=10, 
                          color='yellowgreen', 
                          edgecolor='black', 
                          alpha=0.7)

# Customize the plot
plt.title('Distribution of Alcohol Content in Wines',fontweight='semibold',fontstyle='oblique', fontsize=24,color='magenta')
plt.xlabel('Alcohol Content (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding frequency on top of each bar
for i in range(len(patches)):
    plt.text(patches[i].get_x() + patches[i].get_width()/2,
             patches[i].get_height(),
             f'{int(n[i])}',
             ha='center',
             va='bottom',
             fontsize=10)

# Add mean and median lines
plt.axvline(df['alcohol'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(df['alcohol'].median(), color='blue', linestyle='dashed', linewidth=1, label='Median')

# Add statistical information
stats_text = f'Mean: {df["alcohol"].mean():.2f}\nMedian: {df["alcohol"].median():.2f}\nStd: {df["alcohol"].std():.2f}'
plt.text(0.95, 0.95, stats_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend()
plt.tight_layout()
plt.show()

##4
# Create the boxplot
sns.boxplot(x='quality', 
            y='residual sugar', 
            data=df,
            hue='quality',     
            legend=False,      
            palette='bright')  

# Customize the plot
plt.title('Boxplot of Residual Sugar by Quality Score',fontweight='semibold', fontsize=22,color='slateblue')
plt.xlabel('Quality Score', fontsize=12)
plt.ylabel('Residual Sugar (g/L)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

##5
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a figure with a larger size
plt.figure(figsize=(12, 8))

# Create heatmap
sns.heatmap(correlation_matrix, 
            annot=True,           # Show correlation values
            cmap='coolwarm',      # Color scheme (red for positive, blue for negative)
            center=0,             # Center the colormap at 0
            fmt='.2f',            # Format annotations to 2 decimal places
            square=True,          # Make the plot square-shaped
            linewidths=0.5,       # Add gridlines
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Correlation Matrix of Wine Quality Features', fontweight='semibold',fontsize=24,color='saddlebrown')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

##6
plt.figure(figsize=(10, 6))
sns.lineplot(x='free sulfur dioxide', y='total sulfur dioxide', data=df, marker='s',color='cyan')
plt.title('Trend of Total Sulfur Dioxide Against Free Sulfur Dioxide',fontweight='semibold',fontsize='24',color='forestgreen')
plt.xlabel('Free Sulfur Dioxide (mg/L)')
plt.ylabel('Total Sulfur Dioxide (mg/L)')
plt.grid()

plt.tight_layout()
plt.show()

##Bonus
##1
mean_citric_acid = df['citric acid'].mean()
print(f"Mean citric acid: {mean_citric_acid}")

##2
print()
print('\nMinimum value of each coloumns:')
print(df.min())
print('\nMaximum value of each coloumns:')
print( df.max())

##3
filtered_wines = df[(df['pH'] > 3.5) & (df['alcohol'] > 10)]
average_quality = filtered_wines['quality'].mean()
print(f"\nAverage quality of filtered wines: {average_quality}")

##4
highest_avg_alcohol = df.groupby('quality')['alcohol'].mean().idxmax()
print(f"\nQuality score with highest average alcohol content: {highest_avg_alcohol}")

##5
#.agg(['mean', 'median']
print('Mean')
print(df.groupby('quality').mean())
print('Median')
print(df.groupby('quality').median())

##6
null_citric_acid = df['citric acid'].isnull().sum()
df['citric acid'].fillna(df['citric acid'].median(), inplace=True)
print(f"Number of null values in citric acid column before: {null_citric_acid}")

##7
df.dropna(subset=['residual sugar'], inplace=True)

##8
df['fixed acidity'] = (df['fixed acidity'] - df['fixed acidity'].min()) / (df['fixed acidity'].max() - df['fixed acidity'].min())

##9
df.drop_duplicates(inplace=True)

##10
df['acidity_ratio'] = df['fixed acidity'] / df['volatile acidity']
mean_acidity_ratio = df['acidity_ratio'].mean()
print(f"Mean acidity ratio: {mean_acidity_ratio}")

##11
sns.violinplot(x='quality', y='fixed acidity', data=df)
plt.title('Fixed Acidity vs. Quality')
plt.show()

##12
sns.pairplot(df)
plt.show()

##13
sns.boxplot(y='chlorides', data=df)
plt.title('Chlorides Boxplot for Outlier Detection')
plt.show()

##14
high_chlorides = df[df['chlorides'] > df['chlorides'].quantile(0.95)]
print(high_chlorides)