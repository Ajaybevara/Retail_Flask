from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# ✅ Load model and CSV using your exact path
model = pickle.load(open("best_model.pkl", "rb"))
dataset = pd.read_csv("C:/Users/ajayb/OneDrive/Desktop/AIML_projects/Retail_flask/retail_price.csv")

# Features used for prediction
FEATURES = [
    'qty', 'total_price', 'freight_price', 'product_name_lenght',
    'product_description_lenght', 'product_photos_qty', 'product_weight_g',
    'product_score', 'customers', 'weekday', 'weekend', 'holiday',
    'month', 'year', 's', 'volume', 'comp_1', 'ps1', 'fp1',
    'comp_2', 'ps2', 'fp2', 'comp_3', 'ps3', 'fp3', 'lag_price'
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[f]) for f in FEATURES]
        df_input = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(df_input)[0]

        os.makedirs("plots", exist_ok=True)

        # 📊 Plot 1: Input Feature Values
        plt.figure(figsize=(10, 4))
        sns.barplot(x=FEATURES, y=values, palette="coolwarm")
        plt.xticks(rotation=90)
        plt.title("User Input Feature Values")
        plt.tight_layout()
        plt.savefig("plots/user_input_plot.png")
        plt.close()

        # 📈 Plot 2: Predicted vs Average
        avg_price = dataset["unit_price"].mean()
        plt.figure(figsize=(6, 4))
        sns.barplot(x=["Average Price", "Predicted Price"], y=[avg_price, prediction], palette="Set2")
        plt.title("Predicted vs Average Price")
        plt.ylabel("Unit Price")
        plt.tight_layout()
        plt.savefig("plots/predicted_vs_average.png")
        plt.close()

        # 🏷️ Plot 3: Category Price Comparison
        category_avg = dataset.groupby("product_category_name")["unit_price"].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=category_avg.values, y=category_avg.index, palette="magma")
        plt.title("Top 10 Product Categories by Avg Price")
        plt.xlabel("Average Price")
        plt.tight_layout()
        plt.savefig("plots/avg_price_by_category.png")
        plt.close()

        return render_template('index.html',
                               features=FEATURES,
                               prediction_text=f"Predicted Unit Price: ₹{prediction:.2f}",
                               show_plots=True)
    except Exception as e:
        return render_template('index.html', features=FEATURES,
                               prediction_text=f"Error: {str(e)}")

@app.route('/plots/<filename>')
def plot_image(filename):
    return send_from_directory('plots', filename)

if __name__ == "__main__":
    app.run(debug=True)
