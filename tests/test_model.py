import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.train import prepare,pipeline,metrics,wilson
from src.text import normalize

def test_normalization_preserves_negation():
    assert normalize("This isn't GOOD!!!")=='this is not good'
    assert normalize('GOOD   product 123')=='good product'

def test_remove_duplicate_and_conflicting_text_before_split():
    frame=pd.DataFrame({'Summary':['Good!','good','bad','Bad!','Okay',None],
        'Sentiment':['positive','positive','negative','positive','neutral','positive']})
    cleaned,conflicts=prepare(frame)
    assert conflicts==1
    assert set(cleaned.text)=={'good','okay'}
    assert cleaned.review_hash.nunique()==2

def test_vectorizer_learns_training_vocabulary_only():
    model=pipeline(LogisticRegression())
    model.fit(['excellent lovely product','excellent nice product','awful broken product','awful bad product'],['positive','positive','negative','negative'])
    before=set(model.named_steps['tfidf'].vocabulary_)
    model.predict(['unseentestword awful'])
    assert set(model.named_steps['tfidf'].vocabulary_)==before
    assert 'unseentestword' not in before
    assert model.predict(['excellent lovely'])[0]=='positive'

def test_macro_f1_penalizes_majority_only_predictions():
    result=metrics(['positive','negative','neutral'],['positive']*3)
    assert result['macro_f1']<result['accuracy']
    low,high=wilson(85,100)
    assert low<.85<high
