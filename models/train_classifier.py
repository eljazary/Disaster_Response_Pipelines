import sys
import re 
import nltk
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score




nltk.download('punkt','stopwords', 'wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages_table", engine)
    
    # Create X and Y datasets
    X = df[['message', 'id' ,'original' , 'genre']]
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    
    y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
    df.head()
    
    # Create list containing all category names
    category_names = list(y.columns.values)
    
    return X, y, category_names



import re 
def tokenize(text):
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    words = nltk.word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, f_test, category_names):
    """Returns test accuracy, precision, recall and F1    score for fitted mode   
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """
    # Predict labels for test dataset
    y_pred = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    eval_metrics = get_eval_metrics(np.array(y_test), y_pred, category_names)
    print(eval_metrics)
    metrics = []
    for i in range(len(col_names)):
        
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])


def save_model(model, model_filepath):
    """Pickle fitted model
    
    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    
    Returns:
    None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()