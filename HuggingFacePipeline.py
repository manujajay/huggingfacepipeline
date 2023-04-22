from transformers import pipeline
# The pipeline downloads and caches a default pre-trained model and tokenizer for sentiment analysis
classifier = pipeline('sentiment-analysis') 

# The classifier can then be used on the target text

classifier("Thanks a lot for watching the video. Really appreciate it.")
# {Label: positive, Score: 0.99}

# For more than one sentence, pass a list of sentences to pipeline ()

results = classifier(["Thanks a lot guys", "I hate this video. It's so boring."])

for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
# {Label: positive, Score: 0.99}
# {Label: negative, Score: 0.99}

