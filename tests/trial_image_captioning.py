from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

text = image_to_text("/home/matterj/codebases/eeg_image_reconstruction/output/decoded/eeg_to_image/1/samples/00001_002.png")
print(text)


