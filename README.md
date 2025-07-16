This app is suitable for marketers, analysts, and data science students looking to explore clustering behavior in customer or user datasets.

---

## Features

- Upload any CSV file containing user data
- Automatically handles column cleaning and gender encoding
- Uses a pre-trained KMeans clustering model to segment users
- Interactive visualizations using Plotly for data interpretation
- Option to download the segmented results as a new CSV
- Clean and organized user interface built with Streamlit

---

## Folder Structure

user_segmentation_app/
├── app.py # Main Streamlit application
├── models/
│ └── kmeans_model.pkl # Pre-trained scaler and KMeans model
├── data/
│ └── users.csv # Sample dataset (optional)
├── assets/
│ └── logo.png # Optional logo used in the sidebar
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## Sample Input Format

Your CSV file should include the following columns:

UserID,Age,Gender,Income,SpendingScore,Purchases
1,25,Male,40000,60,3
2,32,Female,50000,40,6
3,28,Female,45000,80,4
...


Column names are flexible — the app will standardize them automatically.
