"""Train on unique public reviews; select on validation, evaluate test once."""
from pathlib import Path
import hashlib
import json
import math
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.text import normalize

ROOT=Path(__file__).resolve().parents[1]
LABELS=['negative','neutral','positive']
SEED=42

def prepare(frame):
    if not {'Summary','Sentiment'}.issubset(frame.columns):
        raise ValueError('Input requires Summary and Sentiment columns')
    valid=frame.loc[frame.Summary.notna() & frame.Sentiment.isin(LABELS),['Summary','Sentiment']].copy()
    valid['text']=valid.Summary.map(normalize)
    valid=valid.loc[valid.text.str.len().ge(3)]
    conflicts=valid.groupby('text').Sentiment.nunique()
    ambiguous=set(conflicts[conflicts.gt(1)].index)
    valid=valid.loc[~valid.text.isin(ambiguous)].drop_duplicates('text')
    valid['review_hash']=valid.text.map(lambda x:hashlib.sha256(x.encode()).hexdigest())
    return valid.sort_values('review_hash').reset_index(drop=True),len(ambiguous)

def pipeline(estimator):
    # Preserve negation. TF-IDF is fit inside each training pipeline only.
    stops=sorted(ENGLISH_STOP_WORDS-{'no','nor','not','never'})
    return Pipeline([
        ('tfidf',TfidfVectorizer(preprocessor=normalize,lowercase=False,
            stop_words=stops,ngram_range=(1,2),min_df=2,max_df=.98,
            max_features=80_000,sublinear_tf=True)),
        ('classifier',estimator)])

def metrics(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)),
        'weighted_f1':float(f1_score(y,pred,labels=LABELS,average='weighted',zero_division=0))}

def wilson(correct,total):
    z=1.959963984540054;p=correct/total
    center=(p+z*z/(2*total))/(1+z*z/total)
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [center-half,center+half]

def run(raw=None,sample_size=45_000):
    raw=Path(raw) if raw else ROOT/'data/raw/Dataset-SA.csv'
    out=ROOT/'data/processed';out.mkdir(parents=True,exist_ok=True)
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True)
    original=pd.read_csv(raw)
    data,conflicts=prepare(original)
    clean_count=len(data)
    if sample_size>len(data):raise ValueError(f'Only {len(data)} unique clean texts; request a smaller --sample-size')
    if sample_size<len(data):
        data,_=train_test_split(data,train_size=sample_size,stratify=data.Sentiment,random_state=SEED)
    train,remainder=train_test_split(data,test_size=.4,stratify=data.Sentiment,random_state=SEED)
    validation,test=train_test_split(remainder,test_size=.5,stratify=remainder.Sentiment,random_state=SEED)
    assert set(train.text).isdisjoint(validation.text) and set(train.text).isdisjoint(test.text) and set(test.text).isdisjoint(validation.text)
    pd.concat([f[['review_hash','Sentiment']].assign(split=name) for name,f in [('train',train),('validation',validation),('test',test)]]).sort_values('review_hash').to_csv(out/'split_manifest.csv',index=False)
    estimators={
        'majority_baseline':DummyClassifier(strategy='most_frequent'),
        'logistic_regression':LogisticRegression(C=2,max_iter=1500,class_weight='balanced',random_state=SEED),
        'linear_svm':LinearSVC(C=1,class_weight='balanced',random_state=SEED,max_iter=5000)}
    candidates={};validation_scores={}
    for name,estimator in estimators.items():
        model=pipeline(estimator)
        model.fit(train.text,train.Sentiment)
        validation_scores[name]=metrics(validation.Sentiment,model.predict(validation.text))
        candidates[name]=model
        print(name,validation_scores[name],flush=True)
    best=max(['logistic_regression','linear_svm'],key=lambda name:validation_scores[name]['macro_f1'])
    # Freeze selection, then refit on train+validation before one test evaluation.
    selected=candidates[best]
    development=pd.concat([train,validation])
    selected.fit(development.text,development.Sentiment)
    predicted=selected.predict(test.text)
    result={
        'source_rows':len(original),'unique_clean_texts':clean_count,'conflicting_texts_removed':conflicts,
        'sampled_unique_reviews':len(data),'train_rows':len(train),'validation_rows':len(validation),'test_rows':len(test),
        'seed':SEED,'labels':LABELS,'validation_scores':validation_scores,
        'selection_metric':'validation macro_f1','selected_model':best,
        'test':metrics(test.Sentiment,predicted),
        'accuracy_95pct_wilson_interval':wilson(int(np.sum(predicted==test.Sentiment.to_numpy())),len(test)),
        'classification_report':classification_report(test.Sentiment,predicted,labels=LABELS,output_dict=True,zero_division=0),
        'test_class_counts':test.Sentiment.value_counts().to_dict(),
        'source_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'python':platform.python_version(),'sklearn':sklearn.__version__,
        'label_provenance':'Publisher used VADER on Summary, followed by manual checking; weak labels, not independent human ground truth.',
        'limitations':['Exact normalized duplicates and conflicting texts removed before split.',
            'Random in-domain split; products and near-duplicate language may occur across splits.',
            'No claim of performance on new products or future reviews.',
            'TF-IDF classification learns publisher sentiment labels; ratings are not features.']}
    (reports/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    # No full review bodies are redistributed in the repository outputs.
    pd.DataFrame({'review_hash':test.review_hash,'actual':test.Sentiment,'predicted':predicted}).to_csv(out/'test_predictions.csv',index=False)
    joblib.dump(selected,artifacts/'sentiment.joblib',compress=3)
    matrix=confusion_matrix(test.Sentiment,predicted,labels=LABELS)
    pd.DataFrame(matrix,index=LABELS,columns=LABELS).to_csv(reports/'confusion_matrix.csv',index_label='actual')
    clf=selected.named_steps['classifier']
    names=selected.named_steps['tfidf'].get_feature_names_out()
    features=[]
    for label,coefficients in zip(clf.classes_,clf.coef_):
        for index in np.argsort(coefficients)[-15:][::-1]:
            features.append({'sentiment':label,'term':names[index],'coefficient':float(coefficients[index])})
    pd.DataFrame(features).to_csv(reports/'top_features.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].imshow(matrix,cmap='Blues')
    axes[0].set_xticks(range(3),LABELS);axes[0].set_yticks(range(3),LABELS)
    axes[0].set_xlabel('Predicted');axes[0].set_ylabel('Actual publisher label')
    axes[0].set_title('Held-out test confusion matrix',loc='left',fontweight='bold')
    for i in range(3):
        for j in range(3):axes[0].text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else '#172554')
    axes[1].barh(list(validation_scores),[v['macro_f1'] for v in validation_scores.values()],color='#0f766e')
    axes[1].set_xlim(0,1);axes[1].set_xlabel('Macro F1')
    axes[1].set_title('Model selection: validation set',loc='left',fontweight='bold')
    fig.savefig(reports/'overview.png',dpi=160);plt.close(fig)
    (reports/'findings.md').write_text(f"""# Findings

Processed {len(original):,} public Flipkart reviews and retained {clean_count:,} unique usable texts after cleaning. Removed {conflicts:,} distinct texts with conflicting sentiment labels. Selected a reproducible stratified sample of {len(data):,} unique texts.

Compared a majority baseline, Logistic Regression, and a linear SVM using TF-IDF unigrams/bigrams. Selected **{best}** by validation macro F1, refit on the 80% development partition, and evaluated once on the untouched {len(test):,}-review test partition.

- Test accuracy: **{result['test']['accuracy']:.2%}**.
- Test macro F1: **{result['test']['macro_f1']:.4f}**.
- Accuracy 95% Wilson interval: {result['accuracy_95pct_wilson_interval'][0]:.2%}–{result['accuracy_95pct_wilson_interval'][1]:.2%}, treating sampled reviews as independent; product clustering may make this optimistic.

The publisher generated labels with VADER and manual checks. This measures agreement with those labels, not independent human judgment. Ratings and product metadata are excluded from model inputs. Exact normalized texts cannot cross partitions; near-duplicates and products can. A product-disjoint or temporal evaluation is a future extension requiring a separate benchmark. Do not substitute an assumed 85% accuracy for the measured result.
""")
    print(json.dumps({k:result[k] for k in ['source_rows','unique_clean_texts','sampled_unique_reviews','selected_model','test']},indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--sample-size',type=int,default=45_000)
    args=p.parse_args();run(args.input,args.sample_size)
