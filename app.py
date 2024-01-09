from flask import Flask, render_template
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
# Sample dataset
data = pd.read_csv(r"C:\Users\DHEERAJ\OneDrive\Desktop\code\Mall_Customers.csv")
# K-means clustering
x = data[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5)
x['Cluster'] = kmeans.fit_predict(x)

# Plotting function
def plot_clusters():
    plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=x['Cluster'], cmap='rainbow')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('K-means Clustering of Mall Customers')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the plot to base64 for displaying in HTML
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_data


@app.route('/')
def index():
    # Render HTML template with the encoded plot
    return render_template('index.html', plot_data=plot_clusters())

if __name__ == '__main__':
    app.run(debug=True, threaded=False)

