# skr-section-labeling
This repository contains files related to section labeling (structure abstract) project.

# Annotated Dataset
"Annotated Data" directory contains files that have 500 manually annotataed abstracts for this project.

# Requirement:
python3
fasttext libraries

# 0.Create and preprocess train, validation and test dataset
python create_sentences_with_context.py
(make sure you have update the data directory and input filenames in the program)

# 1.Find the best hyperparameters for training the model
python find_best_hyperparameters.py
(also need to update the data directory and input filenames in the program)

# 2.Use fasttext to train a model based on the best hyperparameters
(assume you have installed your fasttext binary file here)
./fasttext supervised -input train_filename -output model_filename \
           -dim dimension -wordNgrams wordNgram -epoch epochs
           
# 3.Test model using test dataset
./fasttext test model_filename.bin your_test_filename
