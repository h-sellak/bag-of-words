import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def preprocess(document, punctuation, stop_words):
    """
    Prepares a document to be converted into a specific word representation model.
    
    Args:
        document (str): document to be preprocessed
        punctuation (str): punctuation marks to be removed from the document
        stop_words (list): list of words to be removed from the document 
        
    Returns:
        preprocessed_document (list): final document ready to be converted into a specific word representation model
    """
    
    # Convert to lowercase
    document = document.lower()
    
    # Remove punctuation marks
    document = document.translate(str.maketrans('', '', punctuation))
    
    # Convert document into tokens
    document = document.split()
    
    # Remove stop words
    document = [word for word in document if word not in stop_words]
    
    # Join back words to make a sentence
    document = " ".join(document)
    
    return document

if __name__ == "__main__":
	# Code to be executed when the script is run directly

	# Load documents
    print("\nLoading documents...")
    ## Open the file in read mode
    with open("./data/documents.txt", "r") as file:
	    # Read the contents of the file
        lines = file.readlines()

    # Remove leading/trailing whitespace and create a list of sentences
    documents = [line.strip() for line in lines]

	# Initialise list of pre-processed documents
	#preprocessed_documents = []

	# Punctuation marks
    punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~' # inspired from string.punctuation

	# Stop words
    stop_words = stopwords.words("English") + [str('a')]

	# Preprocessing
    print("\nPreprocessing documents...")
    preprocessed_documents = [preprocess(d, punctuation, stop_words) for d in documents]

	# Print original and pre-processed documents
    print("\nPrinting documents...")
    for i in range(len(documents)):
        print(f"	(before) D{i+1} = {documents[i]}")
        print(f"	(after) D{i+1} = {preprocessed_documents[i]}\n	-----------")
    
    # Create bag of words model
    vectorizer = CountVectorizer()
    bow_model = vectorizer.fit_transform(preprocessed_documents)
    
    # Create a dataframe from a bag of words matrix representation
    bow_to_df = pd.DataFrame(bow_model.toarray(), columns = vectorizer.get_feature_names_out())
    print(bow_to_df)
