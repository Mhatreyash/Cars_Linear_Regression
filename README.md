# Car Fuel Efficiency Prediction 🚗⛽

This project demonstrates how to use **Linear Regression** to predict a car's **Miles per Gallon (MPG)** based on its **Horsepower** and **Weight**.

---

## 📂 Project Structure
```
├── main.py # Main script
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── .gitignore # Git ignore rules
```
---

## ⚙️ Requirements
Install the dependencies with:

```bash
pip install -r requirements.txt
```
▶️ How to Run

Run the script:
```
python main.py
```

This will:
Load and explore the dataset.
Visualize the relationship between Horsepower/Weight and MPG.
Train a Linear Regression model.
Evaluate the model (MSE & R²).
Make a prediction for a sample car (Horsepower=100, Weight=2000).

📊 Example Output
Model Coefficients and Intercept
Mean Squared Error and R² score
Predicted MPG for sample inputs
Scatter plots saved as .png files:
```scatter_hp_mpg.png```
```scatter_weight_mpg.png```

📌 Dataset
The dataset used is mpg.csv (https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv).
---
