import sklearn.feature_extraction
import xml.etree.ElementTree as ET
filename = 'train/00269ea50001a6c699d0222032d45b74b2e7e8be9.None.xml'

tree = ET.parse(filename)
root = tree.getroot()
text = ET.tostring(root)
split_text = text.split("=")
cv = sklearn.feature_extraction.text.CountVectorizer(split_text)
cv.fit_transform(split_text)
vocab = cv.vocabulary_
freq_dict = {key:(vocab[key]/sum(vocab.values())) for key in vocab.keys()}