\# Advanced Image Captioning System with Visual Intelligence and Geo-Inference



\## 📌 Project Overview

This project is an advanced \*\*Image Captioning System\*\* developed using \*\*Deep Learning, Computer Vision, and Visual Intelligence techniques\*\*.  

It generates intelligent, context-aware, and customizable captions for images using a combination of:



\- \*\*CNN-based feature extraction\*\*

\- \*\*LSTM-based sequence generation\*\*

\- \*\*Transformer-enhanced visual-language understanding\*\*

\- \*\*YOLO-based dense region analysis\*\*

\- \*\*Geo-inference and contextual place hypothesis modules\*\*

\- \*\*User personalization and adaptive captioning pipelines\*\*



The system supports both \*\*training and inference\*\* and extends beyond standard image captioning by adding:

\- \*\*emotion-aware captioning\*\*

\- \*\*multi-perspective descriptions\*\*

\- \*\*multilingual caption generation\*\*

\- \*\*hazard/risk-aware visual understanding\*\*

\- \*\*audio narration\*\*

\- \*\*story generation from multiple images\*\*

\- \*\*location inference and geo-reporting\*\*



This project is implemented in \*\*Jupyter Notebook\*\* and includes additional modular support files for advanced visual analysis and reporting.



\---



\## 🚀 Implemented Features (25/25)



1\. \*\*Image Upload Feature\*\*  

2\. \*\*Automatic Caption Generation\*\*  

3\. \*\*CNN-based Feature Extraction\*\*  

4\. \*\*LSTM-based Sentence Generation\*\*  

5\. \*\*Pre-trained Model Support\*\*  

6\. \*\*Real-time Caption Generation\*\*  

7\. \*\*User-friendly Interface\*\*  

8\. \*\*Dataset Training Feature\*\*  

9\. \*\*Caption Accuracy Improvement\*\*  

10\. \*\*Multi-Perspective Captioning\*\*  

11\. \*\*Emotion-Aware Captioning\*\*  

12\. \*\*Personalized Captioning\*\*  

13\. \*\*Personalized User Preference Learning\*\*  

14\. \*\*Audio Caption Narration\*\*  

15\. \*\*Narrative Story Generation from Multiple Images\*\*  

16\. \*\*Multilingual Support\*\*  

17\. \*\*Attention Mechanism Visualization\*\*  

18\. \*\*Dense Region-Level Caption Generation\*\*  

19\. \*\*Hierarchical Multi-Level Caption Generation\*\*  

20\. \*\*Vision-Language Transformer\*\*  

21\. \*\*Reinforcement Learning Optimization\*\*  

22\. \*\*Domain-Adaptive Caption Modes\*\*  

23\. \*\*Risk / Hazard Awareness Captioning\*\*  

24\. \*\*Visual Location Inference Engine\*\*  

25\. \*\*Deep Visual Geo-Inference and Contextual Place Hypothesis Engine v2\*\*



\---



\## 🧠 Core Architecture



\### 1. CNN-based Visual Feature Extraction

\- Uses \*\*pre-trained VGG16 / CNN backbone\*\*

\- Final classification layer removed

\- Extracts high-dimensional visual features for captioning



\### 2. Sequence-based Caption Decoder

\- Uses \*\*Embedding + LSTM-based decoder\*\*

\- Generates captions word-by-word

\- Supports real-time inference on uploaded images



\### 3. Advanced Visual Intelligence Extensions

\- Region-level visual analysis

\- Multi-perspective scene interpretation

\- Emotion-sensitive contextual description

\- Hazard and safety cue detection

\- Location and geo-context hypothesis generation



\### 4. Transformer and Enhanced Language Components

\- Vision-language transformer support

\- Hierarchical caption generation

\- Multilingual / domain-adaptive caption modes

\- Reinforcement-learning-inspired optimization pipeline



\---



\## 📂 Project Structure



Image\_captioning/

│

├── dataset/

│   ├── Flicker8k\_Dataset/

│   ├── Flickr8k\_text/

│   └── uploaded\_test\_images/

│

├── model/

│   ├── features.pkl

│   ├── tokenizer.pkl

│   ├── image\_caption\_model\_final.keras

│   └── image\_caption\_model\_v2.keras

│

└── notebook/

&#x20;   ├── image\_captioing.ipynb

&#x20;   ├── yolov8n.pt

&#x20;   ├── caption\_audio.mp3

&#x20;   ├── geo\_reports/

&#x20;   ├── googlemaps\_reports/

&#x20;   ├── geo\_inference\_log.csv

&#x20;   ├── googlemaps\_geo\_log.csv

&#x20;   ├── logs/

&#x20;   ├── app.py

&#x20;   └── requirements.txt



\---



\## 📊 Dataset Used

\- \*\*Flickr8k Dataset\*\*

\- Includes:

&#x20; - 8,000+ images

&#x20; - Multiple captions per image

\- Main files:

&#x20; - `Flickr8k.token.txt`

&#x20; - `Flickr\_8k.trainImages.txt`



Additional uploaded test images are stored in:

\- `dataset/uploaded\_test\_images/`



\---



\## 🛠️ Technologies Used

\- Python

\- Jupyter Notebook

\- TensorFlow / Keras

\- NumPy

\- Matplotlib

\- Pillow (PIL)

\- ipywidgets

\- Pickle

\- Pandas

\- YOLOv8

\- Deep Learning / NLP / Computer Vision pipelines

\- Streamlit

\- Hugging Face Transformers (BLIP)

\- PyTorch

\- BLIP Image Captioning Model



\---



\## 📦 Installation



This installs all required packages for both the Jupyter Notebook pipeline and the Streamlit-based demo application.



Install dependencies by running the following command in terminal:



pip install -r notebook/requirements.txt



\## ▶️ How to Run

Option 1: Run in Jupyter Notebook

1\.	Open terminal in project folder 

2\.	Start Jupyter Notebook: 

&#x09;	jupyter notebook

3\.	Open: 

&#x09;	notebook/image\_captioing.ipynb

4\.	Run cells step by step 



Option 2: Run Streamlit App

1\. Open terminal in project folder

2\. Run:

&#x09;python -m streamlit run notebook/app.py

3\. Open the local browser URL shown in terminal

4\. Upload image and test real-time caption generation



\## 🔁 Workflow

1\.	Upload image using widget 

2\.	Save uploaded image to dataset/uploaded\_test\_images/ 

3\.	Extract image features using VGG16 

4\.	Load tokenizer and trained model 

5\.	Generate caption word-by-word using LSTM decoder 

6\.	(Optional) Generate audio narration 

7\.	(Optional) Run visual analysis / geo inference modules 



\## 📈 Training Pipeline

•	Load Flickr8k captions 

•	Clean and preprocess captions 

•	Add startseq and endseq 

•	Extract VGG16 features 

•	Create tokenizer 

•	Build CNN + LSTM model 

•	Train using generator 

•	Save: 

&#x09;o features.pkl 

&#x09;o tokenizer.pkl 

&#x09;o trained .keras model 



\## 🔊 Additional Outputs

•	caption\_audio.mp3 → audio narration of generated caption 

•	geo\_reports/ → geo/location report outputs 

•	googlemaps\_reports/ → maps-based reports 

•	geo\_inference\_log.csv → geo inference logs 

•	googlemaps\_geo\_log.csv → maps logs 



\## 📌 Current Status



Implemented:

\- Core image captioning pipeline

\- Training + inference support

\- Image upload and preview

\- Saved trained model and tokenizer

\- Audio narration support

\- Geo-reporting support

\- YOLO-based extension support

\- Advanced visual intelligence support modules integrated



\## 👨‍💻 Author

SAIF CHIKTE



\## 📜 License



This project is intended for academic, educational, and research demonstration purposes only.

"# Image_captioning" 
