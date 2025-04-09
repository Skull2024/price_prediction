# price_prediction
A project for predicting rental prices for real estate using modern machine learning techniques. It includes data collection and preprocessing, training several regression models (Ridge, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, neural network), and interactive forecasting for different cities.

Predicting Rental Prices for Real Estate
This document provides step-by-step instructions for setting up the project, running the code, and obtaining rental price predictions. Follow these instructions to configure the environment, execute the program, and analyze model performance.
________________________________________

1.1. Installation and Running
Clone the repository:
git clone <https or SSH URL>

Navigate to the project folder:
cd price_prediction

o	Ensure that all files (source code, configuration, models, databases) are located in the correct folders.

1.2. Creating a Virtual Environment
python -m venv venv

•	Activate the Virtual Environment:
o	For Windows (PowerShell):
.\venv\Scripts\Activate

o	For Unix/Linux/Mac:
source venv/bin/activate

1.3. Installing Dependencies
•	Install the Required Libraries:
o	Run the following command to install all required packages from the requirements.txt file:

pip install -r requirements.txt

🚀  python main.py

1.4. Configuring Parameters
•	Folder Structure:
o	If needed, rename the folders:
	data – should contain the raw database data.
	src – should contain all the source code.
o	All code files must be placed in the src folder, except for mail.py, config.py, and requirements.txt, which should remain in the project’s root directory.
•	Outlier Removal Settings:
o	Open the config.py file to choose the outlier filtering method.

Method 1 (Default – Quantile Filtering):
o	In the file cleaned_data_v3.py, quantile filtering is set by default (lines 113-116):
	Price: 3rd and 98th percentiles.
	Area: 15th and 98th percentiles.
o	Listings with values outside these percentiles will be removed. You may adjust these threshold percentages (between 0 and 100) as needed.

Method 2 (Threshold Values):
o	In the config.py file, adjust the threshold values on lines 19-23:
	MIN_PRICE, MAX_PRICE, MIN_SQUARE, MAX_SQUARE.
o	Modify the price and area thresholds as necessary.
o	Additionally, in cleaned_data_v3.py, uncomment lines 104-108 and comment out lines 113-116 to switch from quantile filtering.
________________________________________
2. Running the Program

2.1. Starting the Main Script
•	Run the Main Script:
o	From the project root directory, execute:
python main.py

•	City Selection Prompt:
o	The program will display:

Choose a city (Almaty, Astana, Shymkent):
o	Enter the desired city (e.g., Astana) and press Enter.

2.2. Choosing the Mode of Operation
•	Model Usage Prompt:
o	After selecting the city, you will see:
Use the last saved model? (y/n):
o	Enter:
	n if you wish to retrain the models from scratch.
	y if you already have trained models and want to proceed directly to making predictions.

2.3. If Retraining Mode Is Selected (n)
•	Data Cleaning and Preprocessing:
o	The program will clean data from SQLite files, merge tables, and save the cleaned data to a new database.
o	It will then load and process the data (e.g., adding new features, scaling, etc.).
o	The preprocessing pipeline will be saved for future predictions.

•	Training Models:
o	The program will start training regression models, including:
	Ridge, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, and a neural network.
o	It will output performance metrics (MAE for the training set, test set, and cross-validation).
o	All models are saved in a dedicated folder for the selected city. Additionally, a ranking file based on MAE is created in an automatically generated models folder (each city gets its own subfolder).

•	Visualization of Results:
o	The program will generate plots (e.g., scatter plots, error histograms, neural network loss curves) to help analyze model quality.

•	Launching Interactive Prediction:
o	After training and visualization, the program automatically launches the price prediction module (direct_predict.py).

•	Training Classification Models (Optional):
o	In addition to regression models, classification models are also trained to separate listings into high-priced and low-priced categories (this option is under development).

2.4. Interactive Prediction (Available in Both Modes)
The interactive prediction module works in both modes:
•	In retraining mode (n), the above steps will run, and then the prediction module starts.
•	In saved model mode (y), the training stages are skipped, and the prediction module starts immediately (provided that saved training data exists).
•	Entering Apartment Parameters:
o	You will be prompted to enter the following:
	Apartment Area (in m²): Enter a number (e.g., 60) and press Enter.
	Number of Rooms: Enter an integer (e.g., 2) and press Enter.
•	Selecting the Date Mode:
o	Two options will be offered:
1.	Specify a Specific Date:
	Enter the date in the format YYYY-MM-DD (e.g., 2025-04-15).
2.	Specify a Range of Months for the Current Year:
	Enter a range of months (e.g., 4-12 or 4 12). The program will then make a prediction for the 15th day of each month within this range.
•	Selecting Models for Prediction:
o	A list of available models with their MAE rankings will be displayed.
o	Enter the model number or multiple model numbers separated by commas (e.g., 2,3). If multiple models are selected, the arithmetic mean of the predictions will be calculated.

•	Displaying and Saving Results:
o	The program will display a table with predictions containing the following details: date, number of rooms, area, city, and the predicted price.
o	The results will also be saved to a CSV file in the predictions folder (this folder will be automatically created if it does not already exist).

•	Repeat Prediction:
o	The program will ask:
vbnet
Копировать
Would you like to make another prediction? (y/n):
o	Enter y to make another prediction, or n to exit the program.


2.5. Using Saved Models Mode (y)
•	If you select y to use saved models, the program will bypass the training steps and immediately launch the interactive prediction module (as described above).
________________________________________

3. Final Steps
•	Verifying the Results:
o	After the program completes its run, check the CSV files in the predictions folder to review the predictions.
o	Models and preprocessing pipelines are saved in the folders specified in config.py (e.g., in models/<city>).
•	Additional Setup:
o	If you need to fetch new rental price data, refer to the other project available at https://github.com/andprov/krisha.kz. This repository contains a detailed document outlining the steps for downloading, configuring, and running the parser.
________________________________________
This sequence of steps will enable you to properly configure your environment, run the program, obtain rental price predictions, and assess the quality of the models.
