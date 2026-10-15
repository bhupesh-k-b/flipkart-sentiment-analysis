"""Load only locally trained or otherwise trusted joblib models."""
import argparse
from pathlib import Path
import joblib
from src.train import ROOT

def main():
    p=argparse.ArgumentParser(description='Classify a product review')
    p.add_argument('review');p.add_argument('--model',type=Path,default=ROOT/'artifacts/sentiment.joblib')
    args=p.parse_args()
    if not args.review.strip():p.error('Review cannot be empty')
    model=joblib.load(args.model)
    print(model.predict([args.review])[0])

if __name__=='__main__':main()
