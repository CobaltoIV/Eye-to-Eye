
"""
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
"""
import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
signal, fs =torchaudio.load('audio.wav')

# Compute speaker embeddings
embeddings = classifier.encode_batch(signal)

# Perform classification
output_probs, score, index, text_lab = classifier.classify_batch(signal)

# Posterior log probabilities
print(output_probs)

# Score (i.e, max log posteriors)
print(score)

# Index of the predicted speaker
print(index)

# Text label of the predicted speaker
print(text_lab)