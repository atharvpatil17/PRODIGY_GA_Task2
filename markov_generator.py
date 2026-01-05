import random

# Load your text
with open("custom_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into words
words = text.split()

# Build the Markov chain
markov_chain = {}
for i in range(len(words)-1):
    current_word = words[i]
    next_word = words[i+1]
    if current_word not in markov_chain:
        markov_chain[current_word] = []
    markov_chain[current_word].append(next_word)

# Function to generate text
def generate_text(chain, start_word, length=50):
    word = start_word
    output = [word]
    for _ in range(length-1):
        if word in chain:
            word = random.choice(chain[word])
            output.append(word)
        else:
            break
    return " ".join(output)

# Run generator
start_word = random.choice(words)
generated_text = generate_text(markov_chain, start_word, length=100)
print("Generated Text:\n")
print(generated_text)


