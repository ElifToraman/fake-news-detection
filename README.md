Project: Fake News Detection


Project Overview:
This project implements a machine learning model to classify news articles as FAKE or REAL.

Contents of this archive:
- main.py: The main Python script for the project.
- nltk_downloader.py: Script to download NLTK data (if needed).
- data/fake_or_real_news.csv: The dataset used for training and evaluation.
- requirement.txt: Lists all Python libraries needed to run the project.
- FakeNewsDetection_Report.pdf: Your project report.
- README.txt: This file.

Important Note on Data (GoogleNews-vectors-negative300.bin):
The pre-trained Word2Vec model (GoogleNews-vectors-negative300.bin) is approximately 3.64 GB and is too large to be included in this submission archive.

You can download this file from the following link:
https://www.kaggle.com/datasets/adarshsng/googlenewsvectors

Instructions to Run the Project:
1. Extract the 'FakeNewsDetection_Submission.zip' archive.
2. Download the 'GoogleNews-vectors-negative300.bin' file from the link above.
3. Place the downloaded 'GoogleNews-vectors-negative300.bin' file into the 'data/' folder inside the extracted 'FakeNewsDetection_Submission' directory.
4. (Optional) Run 'nltk_downloader.py' if you encounter NLTK data errors.
5. Install required Python libraries: pip install -r requirement.txt
6. Run the main script: python main.py
