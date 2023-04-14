# 🎬 HSE Project Okko
Project in university on creating RecSys for Okko, mentored by <a href="https://github.com/kshurik" target="_blank">Shuhrat Khalilbekov</a> and <a href="https://github.com/5x12" target="_blank">Andrew Wolf</a>.
## ⛏️ Team members
* <a href="https://github.com/missukrof" target="_blank">Anastasiya Kuznetsova</a>
* <a href="https://github.com/aliarahimckulova" target="_blank">Aliya Rahimckulova</a>
* <a href="https://github.com/PBspacey" target="_blank">Nikita Senyatkin</a>
* Tigran Torosyan
# 🔗 Full RecSys Pipeline
Here we have the full pipeline to train and make inference using two-level model architecture
## 📁 Repo Structure
- <a href="https://github.com/missukrof/project-okko-team-work/tree/main/artefacts" target="_blank">artefacts</a> - local storage for models artefacts;
- <a href="https://github.com/missukrof/project-okko-team-work/tree/main/configs" target="_blank">data_prep</a> - data preparation modules to be used during training_pipeline;
- <a href="https://github.com/missukrof/project-okko-team-work/tree/main/models" target="_blank">models</a> - model fit and inference pipeline;
- <a href="https://github.com/missukrof/project-okko-team-work/tree/main/utils" target="_blank">utils</a> - some common functions thatn can be used everywhere.
## ‍💻 Basic files
- <a href="https://github.com/missukrof/project-okko-team-work/blob/main/train.py" target="_blank">train.py</a> - two-stage model training (the first level - LightFM, the second - CatBoost classifier);
- <a href="https://github.com/missukrof/project-okko-team-work/blob/main/inference.py" target="_blank">inference.py</a> - get recommendations from two-stage model for a particular user.
