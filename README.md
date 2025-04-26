# Introduction 
In this project I have implemented a content based recommendation system based on multi_category products and electronic products. Three different approaches were used fort he recommendation system, the TF-IDF and cosine similarity approach and machine learning algorithms like Random Forest and XGBoost approach. 


## Getting Started
### Prerequisites

- Python 3.7+
- pip (Python package Installer)

### Clone the Repository
```bash
git clone https://github.com/AnshulPatil2911/Content-based-recommendation-system.git
```

### Install the required packages

   ```bash
   pip install -r requirements.txt
   ```

### Running the notebook to train the models
1. Navigate to the Data directory

   ```bash
   cd project/Data
   ```
2. Load all the csv files from the drive path provided in the data_path text file

3. The csv files in the drive are preprocessed file, store them in a directory, please maintain the following folder structure
   
   ![image](https://github.com/user-attachments/assets/48267aa4-06fa-46a7-adc1-2931b404c41f)

4. Now you can directly run the Feature_engineering.ipynb in order to get the personalized recommendations

5. Before running the notebook ensure that the csv path are correctly provided in the notebook

### Running the streamlit app

1. Navigate to the Data directory

   ```bash
   cd project/Notebook
   ```
2. Ensure that the pkl files generated after running the Feature_engineering.ipynb is in the same directory as the app.py.

3. Run the app.py to launch the streamlit application(recommendation system dashboard)
    ```bash
   python app.py 
   ```

### Accessing the report files

1. In order to view the final report navigate to the Report directory
   ```bash
   cd project/Report
   ```
2. The report (Final_report.pdf) contains the step followed for recommendations for all the approaches, feature engineering, feature selection, model training, results interpretation, test evaluation and tool explanation

### Accessing the video demonstration
1. In order to view the final report navigate to the Report directory
   ```bash
   cd project/Video_demonstration
   ```
