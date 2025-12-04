#!/bin/bash

# HRM Language Dataset Preparation Script
# Creates a comprehensive character-level training dataset

echo "🎯 HRM Language Dataset Preparation"
echo "==================================="

# Create dataset directory structure
mkdir -p data/text/{raw,processed,training,validation,test}

echo "📁 Created dataset directory structure"

# Download and prepare diverse text corpora for character-level training
# Focus on content that promotes reasoning, conversation, and knowledge

echo "📥 Downloading training corpora..."

# 1. OpenWebText2 (diverse web content)
echo "Downloading OpenWebText2 sample..."
curl -L "https://huggingface.co/datasets/openwebtext/resolve/main/plain_text/train-00000-of-00001.parquet" -o data/text/raw/openwebtext_sample.parquet 2>/dev/null || echo "OpenWebText download failed, will use local data"

# 2. Books corpus (literary reasoning)
echo "Setting up books corpus..."
mkdir -p data/text/raw/books
cat > data/text/raw/books/sample_books.txt << 'EOF'
Pride and Prejudice by Jane Austen

It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.

However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters.

"My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?"

Mr. Bennet replied that he had not.

"But it is," returned she; "for Mrs. Long has just been here, and she told me all about it."

Mr. Bennet made no answer.

"Do you not want to know who has taken it?" cried his wife impatiently.

"You want to tell me, and I have no objection to hearing it."

This was invitation enough.

"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it, that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week."

"What is his name?"

"Bingley."

"Is he married or single?"

"Oh! Single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!"

"How so? How can it affect them?"

"My dear Mr. Bennet," replied his wife, "how can you be so tiresome! You must know that I am thinking of his marrying one of them."

"Is that his design in settling here?"

"Design! Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes."

"I see no occasion for that. You and the girls may go, or you may send them by themselves, which perhaps will be still better, for as you are as handsome as any of them, Mr. Bingley may like you the best of the party."

"My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty."

"In such cases, a woman has not often much beauty to think of."

"But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood."

"It is more than I engage for, I assure you."

"But consider your daughters. Only think what an establishment it will be for one of them. Sir William and Lady Lucas are determined to go, merely on that account, for in general, you know, they visit no newcomers. Indeed you must go, for it will be impossible for us to visit him if you do not."

"You are over-scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls; though I must throw in a good word for my little Lizzy."

"I desire you will do no such thing. Lizzy is not a bit better than the others; and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference."

"They have none of them much to recommend them," replied he; "they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters."

"Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion for my poor nerves."

"You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these last twenty years at least."
EOF

# 3. Code corpus (programming reasoning)
echo "Setting up code corpus..."
mkdir -p data/text/raw/code
cat > data/text/raw/code/sample_code.py << 'EOF'
# Python examples for reasoning and problem solving

def binary_search(arr, target):
    """
    Binary search algorithm - demonstrates algorithmic reasoning
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

def fibonacci_recursive(n):
    """
    Fibonacci sequence - recursive reasoning
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n):
    """
    Fibonacci sequence - iterative approach
    More efficient for large n due to reasoning about computational complexity
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(num):
    """
    Primality testing - mathematical reasoning
    """
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False

    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6

    return True

def gcd(a, b):
    """
    Greatest Common Divisor - Euclidean algorithm
    Demonstrates mathematical reasoning and recursion
    """
    while b != 0:
        a, b = b, a % b
    return a

def sieve_of_eratosthenes(n):
    """
    Sieve of Eratosthenes - efficient prime generation
    Algorithmic reasoning for optimization
    """
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False

    for i in range(2, n+1):
        if is_prime[i]:
            primes.append(i)

    return primes

# Example usage and reasoning
if __name__ == "__main__":
    # Binary search demonstration
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    result = binary_search(arr, target)
    print(f"Binary search: {target} found at index {result}")

    # Fibonacci comparison
    n = 10
    recursive_result = fibonacci_recursive(n)
    iterative_result = fibonacci_iterative(n)
    print(f"Fibonacci {n}: recursive={recursive_result}, iterative={iterative_result}")

    # Prime checking
    test_numbers = [2, 3, 4, 5, 16, 17, 18, 19, 23]
    for num in test_numbers:
        print(f"{num} is {'prime' if is_prime(num) else 'not prime'}")

    # GCD calculation
    print(f"GCD of 48 and 18 is {gcd(48, 18)}")

    # Prime generation
    primes = sieve_of_eratosthenes(50)
    print(f"Primes up to 50: {primes}")
EOF

# 4. Conversation corpus (dialogue reasoning)
echo "Setting up conversation corpus..."
mkdir -p data/text/raw/conversations
cat > data/text/raw/conversations/sample_conversations.txt << 'EOF'
Human: Can you explain how photosynthesis works?

Assistant: Photosynthesis is the process by which plants convert light energy into chemical energy. Let me break this down step by step:

1. Light absorption: Chlorophyll in plant leaves absorbs sunlight, primarily in the blue and red wavelengths.

2. Water uptake: Plants absorb water from the soil through their roots.

3. Carbon dioxide intake: Plants take in CO2 from the atmosphere through small openings called stomata in their leaves.

4. Chemical reaction: Using the absorbed light energy, plants convert water and carbon dioxide into glucose (a sugar) and oxygen.

The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2

This process occurs in specialized organelles called chloroplasts, which contain the chlorophyll pigments. The light-dependent reactions happen in the thylakoid membranes, while the light-independent reactions (Calvin cycle) occur in the stroma.

Photosynthesis is crucial for life on Earth because it:
- Produces oxygen that animals breathe
- Creates the base of the food chain
- Helps regulate atmospheric CO2 levels
- Provides renewable energy storage in biomass

Human: That's very clear! Can you explain why plants are green?

Assistant: Great question! Plants appear green because of how they interact with light. Here's the reasoning:

1. Chlorophyll absorbs light energy for photosynthesis
2. Chlorophyll primarily absorbs blue and red wavelengths of light
3. Green light is reflected, not absorbed
4. Our eyes perceive the reflected green light as the plant's color

The chlorophyll molecule has a specific structure that allows it to capture light energy efficiently. There are actually several types of chlorophyll (a, b, c, d) with slightly different absorption spectra, but they all reflect green light.

This green reflection is actually beneficial for plants because:
- Green light penetrates deeper into leaf layers
- It allows photosynthesis to occur throughout the leaf
- It provides visual cues for pollinators and seed dispersers

Some plants have evolved other pigments like anthocyanins (red/purple) or carotenoids (yellow/orange) that mask the green chlorophyll, especially in autumn when chlorophyll breaks down.

Human: How does evolution work?

Assistant: Evolution is the process by which species change over time through natural selection. Let me explain the key mechanisms:

1. Genetic Variation: All populations have genetic diversity due to mutations, sexual reproduction, and genetic recombination.

2. Natural Selection: Individuals with traits better suited to their environment are more likely to survive and reproduce. This is often called "survival of the fittest," but it's really about reproductive success.

3. Heredity: Beneficial traits are passed to offspring through genes.

4. Time: Evolution requires many generations to produce significant changes.

Key mechanisms of evolution:
- Natural selection (adaptation to environment)
- Genetic drift (random changes in small populations)
- Gene flow (movement of genes between populations)
- Mutations (source of new genetic variation)

Evidence for evolution includes:
- Fossil record showing transitional forms
- Homologous structures in different species
- Vestigial organs (structures with no current function)
- Embryological development patterns
- Molecular biology (DNA similarities between species)
- Observed evolution in laboratory settings and nature

Evolution explains both the diversity of life and the adaptations we see in organisms. It's a scientific theory supported by overwhelming evidence from multiple fields of study.

Human: What's the difference between weather and climate?

Assistant: Excellent question! Many people confuse these terms, but they're quite different:

Weather:
- Short-term atmospheric conditions
- What you experience day-to-day
- Changes rapidly (hours to days)
- Local or regional scope
- Examples: Today's temperature, chance of rain, wind speed

Climate:
- Long-term patterns and averages
- Statistical description of weather over time
- Takes years or decades to change significantly
- Regional to global scope
- Examples: Average temperature for a region, typical precipitation patterns

Key differences:
1. Time scale: Weather is immediate, climate is long-term
2. Scope: Weather is local, climate is broader
3. Predictability: Weather is hard to predict beyond a few days, climate patterns are more stable
4. Causes: Weather is caused by short-term atmospheric dynamics, climate is influenced by geography, ocean currents, and human activities

Climate change refers to long-term shifts in climate patterns, often due to human activities like burning fossil fuels, which increase greenhouse gases in the atmosphere.

Understanding this distinction helps us better communicate about environmental issues and weather forecasting.
EOF

# 5. Reasoning corpus (logical thinking)
echo "Setting up reasoning corpus..."
mkdir -p data/text/raw/reasoning
cat > data/text/raw/reasoning/sample_reasoning.txt << 'EOF'
Logic and Reasoning Examples

Syllogisms - Deductive Reasoning:

All men are mortal.
Socrates is a man.
Therefore, Socrates is mortal.

This is a valid syllogism with the following structure:
Major premise: All A are B
Minor premise: C is A
Conclusion: Therefore, C is B

The reasoning is sound because if both premises are true, the conclusion must be true.

Inductive Reasoning:

The sun has risen every day for thousands of years.
Therefore, the sun will rise tomorrow.

This is inductive reasoning because it moves from specific observations to general conclusions. While the conclusion is likely true, it's not guaranteed - it's based on probability rather than certainty.

Abductive Reasoning:

The grass is wet.
If it rained, the grass would be wet.
Therefore, it probably rained.

This is abductive reasoning, which seeks the most likely explanation for observed facts. It's commonly used in scientific discovery and medical diagnosis.

Critical Thinking Skills:

1. Identify assumptions: What is taken for granted in an argument?
2. Evaluate evidence: Is the evidence relevant, sufficient, and reliable?
3. Recognize biases: Are there personal, cultural, or logical biases affecting the reasoning?
4. Consider alternatives: Are there other explanations or viewpoints?
5. Draw logical conclusions: Does the evidence support the conclusion?

Problem-Solving Strategies:

1. Define the problem clearly
2. Gather relevant information
3. Generate multiple solutions
4. Evaluate options systematically
5. Implement the best solution
6. Review and learn from the outcome

Scientific Method:

1. Observe a phenomenon
2. Form a hypothesis
3. Design experiments to test the hypothesis
4. Collect and analyze data
5. Draw conclusions based on evidence
6. Share findings and allow peer review
7. Refine or reject the hypothesis based on new evidence

Mathematical Reasoning:

Theorem: In a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.

Proof by contradiction:
Assume the theorem is false. Then either:
- a² + b² > c², or
- a² + b² < c²

But geometric constructions show that both cases lead to contradictions. Therefore, the theorem must be true.

This demonstrates deductive reasoning in mathematics, where conclusions are proven with absolute certainty from given axioms.

Ethical Reasoning:

When faced with a moral dilemma, consider:

1. Consequences: What outcomes will result from each action?
2. Duties: What obligations do you have to others?
3. Character: What would a virtuous person do?
4. Rights: Whose rights are affected?
5. Fairness: Is the solution equitable?

Ethical frameworks:
- Utilitarianism: Maximize overall happiness
- Deontology: Follow moral rules regardless of consequences
- Virtue ethics: Cultivate good character traits
- Care ethics: Prioritize relationships and compassion

These reasoning frameworks help us make better decisions in complex situations.
EOF

# 6. Scientific corpus (evidence-based reasoning)
echo "Setting up scientific corpus..."
mkdir -p data/text/raw/science
cat > data/text/raw/science/sample_science.txt << 'EOF'
Scientific Reasoning and Methodology

The scientific method provides a systematic approach to understanding the natural world through empirical evidence and logical reasoning.

Key Principles:

1. Empiricism: Knowledge comes from observation and experimentation
2. Objectivity: Conclusions should be free from personal bias
3. Reproducibility: Results should be repeatable by others
4. Parsimony: The simplest explanation is usually best (Occam's Razor)
5. Skepticism: Claims require sufficient evidence

Experimental Design:

A well-designed experiment includes:
- Control group: Baseline for comparison
- Experimental group: Receives the treatment
- Randomization: Reduces bias in group assignment
- Replication: Multiple trials for statistical validity
- Placebo controls: Accounts for psychological effects

Statistical Reasoning:

Understanding p-values:
- p < 0.05: Result unlikely due to chance (95% confidence)
- p < 0.01: Result very unlikely due to chance (99% confidence)
- p < 0.001: Result extremely unlikely due to chance (99.9% confidence)

However, statistical significance doesn't prove practical importance. Consider:
- Effect size: How large is the observed effect?
- Confidence intervals: Range of plausible values
- Sample size: Larger samples provide more reliable results
- Study design: Was the methodology appropriate?

Correlation vs. Causation:

Correlation: Two variables tend to change together
Causation: One variable directly causes changes in another

Common fallacies:
- Post hoc ergo propter hoc: "After this, therefore because of this"
- Cum hoc ergo propter hoc: "With this, therefore because of this"
- Confounding variables: Unmeasured factors affecting both variables

Establishing causation requires:
- Temporal precedence: Cause precedes effect
- Covariation: Variables change together
- Elimination of alternatives: No other explanations

Falsifiability:

A scientific hypothesis must be testable and potentially disprovable. Theories that explain everything explain nothing.

Examples:
- "Gravity exists" - Falsifiable (we could find places where it doesn't work)
- "God exists" - Not falsifiable (no test can disprove it)
- "All swans are white" - Falsifiable (finding one black swan disproves it)

Paradigm Shifts:

Scientific revolutions occur when anomalies accumulate against existing theories, leading to new paradigms:

- Copernican Revolution: Heliocentric model replaced geocentric
- Darwinian Revolution: Evolution by natural selection
- Quantum Revolution: Probabilistic nature of matter
- Plate Tectonics: Continental drift explained

These shifts require not just new evidence, but a willingness to reconsider fundamental assumptions.

Peer Review and Replication:

Scientific knowledge advances through:
- Peer review: Expert evaluation before publication
- Replication: Independent verification of results
- Meta-analysis: Statistical combination of multiple studies
- Open science: Sharing data and methodologies

This rigorous process helps ensure scientific conclusions are reliable and accurate.
EOF

echo "✅ Downloaded/created sample corpora"

# Process and combine datasets
echo "🔄 Processing datasets..."

# Combine all text files into a single training corpus
cat data/text/raw/books/sample_books.txt > data/text/processed/training_corpus.txt
echo "" >> data/text/processed/training_corpus.txt
cat data/text/raw/code/sample_code.py >> data/text/processed/training_corpus.txt
echo "" >> data/text/processed/training_corpus.txt
cat data/text/raw/conversations/sample_conversations.txt >> data/text/processed/training_corpus.txt
echo "" >> data/text/processed/training_corpus.txt
cat data/text/raw/reasoning/sample_reasoning.txt >> data/text/processed/training_corpus.txt
echo "" >> data/text/processed/training_corpus.txt
cat data/text/raw/science/sample_science.txt >> data/text/processed/training_corpus.txt

echo "✅ Combined training corpus created"

# Create validation set (10% of data)
head -n $(($(wc -l < data/text/processed/training_corpus.txt) / 10)) data/text/processed/training_corpus.txt > data/text/processed/validation_corpus.txt

echo "✅ Validation corpus created"

# Create test set (5% of data)
head -n $(($(wc -l < data/text/processed/training_corpus.txt) / 20)) data/text/processed/training_corpus.txt > data/text/processed/test_corpus.txt

echo "✅ Test corpus created"

# Generate statistics
echo "📊 Dataset Statistics:"
echo "Training samples: $(wc -l < data/text/processed/training_corpus.txt)"
echo "Validation samples: $(wc -l < data/text/processed/validation_corpus.txt)"
echo "Test samples: $(wc -l < data/text/processed/test_corpus.txt)"
echo "Total characters: $(wc -c < data/text/processed/training_corpus.txt)"

echo ""
echo "🎯 HRM Language Dataset Ready!"
echo ""
echo "Dataset includes:"
echo "├── 📚 Literary works (Pride & Prejudice)"
echo "├── 💻 Programming examples (Python algorithms)"
echo "├── 💬 Conversations (educational dialogues)"
echo "├── 🧠 Reasoning exercises (logic, critical thinking)"
echo "└── 🔬 Scientific content (methodology, evidence)"
echo ""
echo "This diverse corpus will help HRM develop:"
echo "• Conversational abilities"
echo "• Logical reasoning skills"
echo "• Programming understanding"
echo "• Scientific literacy"
echo "• Literary comprehension"
echo ""
echo "Next steps:"
echo "1. Run: python3 scripts/train_character_language.py"
echo "2. Monitor training progress in logs/"
echo "3. Test generation: ./hrm_system --generate-text"
echo "4. Expand dataset with more diverse content"
echo ""
echo "The HRM is ready to begin character-level language training! 🚀"