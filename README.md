### LiriKlas
by Gabriel Aldrich S. Calubayan, Nina Patricia Sapitula, and Kaila Ondoy in partial fulfillment of CS 180 of UP Diliman Department of Computer Science

## Relevant links
* Project proposal: https://drive.google.com/file/d/1DK5hbnfoKyiLfj_nhCwITaDrExvWSa-P/view?usp=sharing
* Github repo here: https://github.com/gabrielcalubayan72/cs180-23b-genreclassifier
* Access the deployed website here: https://liriklas-9f656410ec11.herokuapp.com/

## Relevant files/notebooks  in this repository
* `song_lyrics.csv`

The raw dataset of Genius song lyrics from Kaggle (https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/data) is not included in the repository because is too large (more than (9 gigabytes).

* `preprocessing.ipynb`

The raw data is then processed to filter null values, and only include song entries with language tags `fil` and `ceb`. The partially processed dataset is exported in `filceb_lyrics.csv`

* `nlp.ipynb`

Additional preprocessing and cleaning of `filceb_lyrics.csv` is carried out, such as stopword removal, format consistency of lyrics (e.g., removal of punctuations, song markers), and some data visualization. The final processed lyrics dataset is exported in `processed_lyrics.ipynb`

* `model.ipynb`

This is where the Naive Bayes model is built by splitting `processed_lyrics.ipynb` into stratified training and test sets. Further tuning is also done via `GridSearchCV` attaining an accuracy of about `76%`. 

* `./web-app`

A web app (Flask) then takes  in the model and creates a way for users to input text lyrics and showing the classification result of the model.
