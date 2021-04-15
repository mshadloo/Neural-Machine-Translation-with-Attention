wget http://www.manythings.org/anki/fra-eng.zip
unzip fra-eng.zip
rm fra-eng.zip
mkdir data
mv fra.txt ./data
wget https://nlp.stanford.edu/data/glove.6B.zip
mkdir embedding
unzip glove.6B.zip -d ./embedding
rm glove.6B.zip
