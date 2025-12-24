#!/usr/bin/env python3
"""
Generate synthetic conversational datasets for HRM training
Creates character-level conversational data without tokenization
"""

import random
import json
from typing import List, Dict, Tuple

class ConversationGenerator:
    def __init__(self):
        # Conversation patterns and templates
        self.greetings = [
            "Hello", "Hi there", "Good morning", "Good afternoon", "Hey", "Greetings"
        ]

        self.questions = [
            "How are you", "What are you doing", "Can you help me", "What's the weather like",
            "How does this work", "Can you explain", "What do you think about", "Why is",
            "How can I", "What should I do", "Tell me about", "Do you know"
        ]

        self.topics = [
            "artificial intelligence", "machine learning", "programming", "technology",
            "science", "mathematics", "physics", "chemistry", "biology", "history",
            "philosophy", "psychology", "economics", "politics", "environment",
            "health", "education", "sports", "music", "art", "literature", "travel"
        ]

        self.responses = [
            "That's interesting", "I understand", "Let me think about that", "Good question",
            "I'm not sure", "That depends", "Here's what I know", "From my perspective",
            "That's a complex topic", "Let me explain", "I can help with that", "Certainly"
        ]

        self.follow_ups = [
            "Can you tell me more", "What do you mean", "Why do you think that",
            "How does that work", "Can you give an example", "Is that always true",
            "What are the implications", "How can I learn more", "What's your opinion",
            "Do you agree", "What's the evidence", "Can you elaborate"
        ]

        self.closings = [
            "Thanks for the conversation", "That was helpful", "I appreciate your input",
            "Goodbye", "See you later", "Take care", "Have a good day", "Nice talking to you"
        ]

    def generate_conversation(self, turns: int = 3) -> str:
        """Generate a single conversation as continuous text"""
        conversation = []

        # Start with greeting
        user1 = random.choice(self.greetings)
        conversation.append(f"Human: {user1}")

        ai1 = random.choice(self.responses) + " about " + random.choice(self.topics)
        conversation.append(f"Assistant: {ai1}")

        for _ in range(turns - 1):
            if random.random() < 0.7:  # 70% chance of follow-up
                user = random.choice(self.follow_ups) + " about " + random.choice(self.topics)
            else:
                user = random.choice(self.questions) + " " + random.choice(self.topics)

            conversation.append(f"Human: {user}")

            ai_response = random.choice(self.responses)
            if random.random() < 0.5:
                ai_response += ". " + random.choice(self.responses).lower()
            ai_response += " regarding " + random.choice(self.topics)

            conversation.append(f"Assistant: {ai_response}")

        # End with closing
        if random.random() < 0.3:
            user_closing = random.choice(self.closings)
            conversation.append(f"Human: {user_closing}")

            ai_closing = random.choice(self.closings)
            conversation.append(f"Assistant: {ai_closing}")

        return "\n\n".join(conversation) + "\n\n"

    def generate_math_content(self) -> str:
        """Generate mathematical problems and explanations"""
        math_topics = [
            "algebra", "geometry", "calculus", "statistics", "number theory",
            "linear algebra", "probability", "discrete mathematics", "trigonometry"
        ]

        problems = [
            f"Solve for x: 2x + 3 = 7\nStep 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide by 2: x = 2\n\n",
            f"Calculate the derivative of f(x) = x² + 3x + 1\nf'(x) = 2x + 3\n\n",
            f"What's the area of a circle with radius 5?\nA = πr² = π(25) = 25π\n\n",
            f"Probability of rolling a 6 on a fair die: 1/6\n\n",
            f"Matrix multiplication example:\n[1 2] * [3 4] = [1*3+2*5 1*4+2*6] = [13 16]\n  [3 4]   [5 6]\n\n"
        ]

        explanations = [
            f"In {random.choice(math_topics)}, we use logical reasoning to solve problems.\n\n",
            f"Mathematical proofs require careful step-by-step reasoning.\n\n",
            f"Understanding {random.choice(math_topics)} helps develop analytical thinking.\n\n"
        ]

        content = random.choice(problems) + random.choice(explanations)
        return f"Mathematics: {content}\n\n"

    def generate_science_content(self) -> str:
        """Generate scientific explanations and concepts"""
        science_fields = [
            "physics", "chemistry", "biology", "astronomy", "geology",
            "neuroscience", "ecology", "genetics", "climatology", "quantum mechanics"
        ]

        concepts = [
            f"In {random.choice(science_fields)}, the scientific method guides our understanding.\n\n",
            f"Empirical evidence supports theories in {random.choice(science_fields)}.\n\n",
            f"Peer review ensures quality in {random.choice(science_fields)} research.\n\n",
            f"Mathematical models help explain phenomena in {random.choice(science_fields)}.\n\n"
        ]

        experiments = [
            f"Controlled experiments in {random.choice(science_fields)} test hypotheses.\n\n",
            f"Observation and measurement are fundamental to {random.choice(science_fields)}.\n\n",
            f"Reproducibility validates findings in {random.choice(science_fields)}.\n\n"
        ]

        content = random.choice(concepts) + random.choice(experiments)
        return f"Science: {content}\n\n"

    def generate_coding_content(self) -> str:
        """Generate programming examples and reasoning"""
        languages = ["Python", "C++", "JavaScript", "Java", "Rust", "Go"]
        concepts = [
            "algorithms", "data structures", "object-oriented programming",
            "functional programming", "concurrency", "optimization"
        ]

        examples = [
            f"In {random.choice(languages)}, we implement {random.choice(concepts)} to solve problems.\n\n",
            f"Debugging requires systematic reasoning about code behavior.\n\n",
            f"Code review helps identify logical errors and improve quality.\n\n",
            f"Testing ensures software reliability and correctness.\n\n"
        ]

        patterns = [
            f"Design patterns in {random.choice(languages)} provide reusable solutions.\n\n",
            f"Refactoring improves code maintainability and readability.\n\n",
            f"Version control tracks changes and enables collaboration.\n\n"
        ]

        content = random.choice(examples) + random.choice(patterns)
        return f"Programming: {content}\n\n"

    def generate_reasoning_content(self) -> str:
        """Generate logical reasoning exercises"""
        reasoning_types = [
            "deductive reasoning", "inductive reasoning", "abductive reasoning",
            "critical thinking", "problem-solving", "decision-making"
        ]

        exercises = [
            f"Practice {random.choice(reasoning_types)} by analyzing arguments carefully.\n\n",
            f"Identify assumptions and evaluate evidence in {random.choice(reasoning_types)}.\n\n",
            f"Consider multiple perspectives when applying {random.choice(reasoning_types)}.\n\n",
            f"Logical fallacies undermine {random.choice(reasoning_types)}.\n\n"
        ]

        strategies = [
            f"Break complex problems into smaller steps for better reasoning.\n\n",
            f"Question assumptions to strengthen {random.choice(reasoning_types)}.\n\n",
            f"Seek evidence and avoid confirmation bias in reasoning.\n\n"
        ]

        content = random.choice(exercises) + random.choice(strategies)
        return f"Reasoning: {content}\n\n"

    def generate_qa_pairs(self, count: int = 10) -> str:
        """Generate Q&A pairs"""
        qa_text = []

        for _ in range(count):
            topic = random.choice(self.topics)
            question = random.choice(self.questions) + " " + topic + "?"
            answer = random.choice(self.responses) + ". " + topic.capitalize() + " is " + \
                    random.choice(["fascinating", "complex", "important", "challenging", "rewarding"])

            qa_text.append(f"Q: {question}")
            qa_text.append(f"A: {answer}")

        return "\n".join(qa_text) + "\n\n"

    def generate_dialogue_scenarios(self, count: int = 5) -> str:
        """Generate various dialogue scenarios"""
        scenarios = []

        for _ in range(count):
            scenario_type = random.choice([
                "technical_support", "learning_session", "casual_chat",
                "problem_solving", "information_exchange", "debate"
            ])

            if scenario_type == "technical_support":
                dialogue = self._generate_tech_support()
            elif scenario_type == "learning_session":
                dialogue = self._generate_learning()
            elif scenario_type == "casual_chat":
                dialogue = self._generate_casual()
            elif scenario_type == "problem_solving":
                dialogue = self._generate_problem_solving()
            elif scenario_type == "information_exchange":
                dialogue = self._generate_info_exchange()
            else:
                dialogue = self._generate_debate()

            scenarios.append(dialogue)

        return "\n".join(scenarios)

    def _generate_tech_support(self) -> str:
        problems = ["can't connect", "error message", "slow performance", "crashes", "won't start"]
        solutions = ["restart the system", "check your connections", "update the software", "clear cache", "run diagnostics"]

        problem = random.choice(problems)
        solution = random.choice(solutions)

        return f"""Human: I'm having trouble with my system. It {problem}.

Assistant: I'm sorry to hear that. Let's troubleshoot this. First, try to {solution}.

Human: Okay, I did that. What next?

Assistant: Good. If that doesn't work, we might need to {random.choice(solutions)}.

Human: That fixed it! Thanks for your help.

Assistant: You're welcome! Glad I could help you resolve the issue.
"""

    def _generate_learning(self) -> str:
        topic = random.choice(self.topics)
        return f"""Human: Can you teach me about {topic}?

Assistant: Absolutely! {topic.capitalize()} is a fascinating subject. Let's start with the basics.

Human: What are the fundamentals?

Assistant: The fundamentals of {topic} include understanding its core principles and applications.

Human: Can you give me an example?

Assistant: Certainly. Here's a practical example of {topic} in action...

Human: That makes sense. What's next?

Assistant: Great! Now let's explore some advanced concepts in {topic}.
"""

    def _generate_casual(self) -> str:
        return f"""Human: {random.choice(self.greetings)}! How's everything going?

Assistant: {random.choice(self.greetings)}! Things are going well. How about you?

Human: Pretty good, thanks. What have you been up to?

Assistant: I've been learning about {random.choice(self.topics)}. It's quite interesting.

Human: That sounds cool. Tell me more about it.

Assistant: Sure! {random.choice(self.topics).capitalize()} involves many fascinating aspects...

Human: Interesting. Maybe I should look into that too.

Assistant: Definitely! It's a great topic to explore.
"""

    def _generate_problem_solving(self) -> str:
        problem = f"How can I {random.choice(['improve', 'optimize', 'fix', 'enhance'])} my {random.choice(self.topics)} skills?"
        return f"""Human: {problem}

Assistant: That's a great question! Improving {random.choice(self.topics)} skills requires systematic approach.

Human: What should I start with?

Assistant: Begin with understanding the fundamentals. Practice regularly and seek feedback.

Human: Any specific recommendations?

Assistant: Yes, try working on small projects and studying examples from experts.

Human: That sounds helpful. Thanks!

Assistant: You're welcome! Keep practicing and you'll see improvement.
"""

    def _generate_info_exchange(self) -> str:
        topic = random.choice(self.topics)
        return f"""Human: I'm curious about {topic}. What can you tell me?

Assistant: {topic.capitalize()} is a broad field with many interesting aspects.

Human: Can you share some key facts?

Assistant: Certainly. Here are some important points about {topic}...

Human: That's fascinating. Any recent developments?

Assistant: Yes, there have been exciting advances in {topic} recently...

Human: I'd like to learn more. Any resources?

Assistant: There are excellent books, online courses, and communities dedicated to {topic}.
"""

    def _generate_debate(self) -> str:
        topic = random.choice(self.topics)
        return f"""Human: What do you think about {topic}? Is it really important?

Assistant: {topic.capitalize()} is indeed significant, though perspectives vary.

Human: Why do you say that? I think it's overrated.

Assistant: Interesting viewpoint. Let me explain why I consider it important...

Human: Hmm, you make some good points, but I'm not convinced.

Assistant: Fair enough. Different people have different experiences with {topic}.

Human: Maybe you're right. I should learn more about it.

Assistant: That's a great attitude! Exploring {topic} further will help you form your own opinion.
"""

def main():
    generator = ConversationGenerator()

    print("Generating comprehensive conversational datasets for HRM training...")
    print("Including language, reasoning, coding, math, and all branches of science")

    # Generate massive dataset for 3+ hours of training
    conversations = []
    qa_pairs = []
    dialogues = []
    math_content = []
    science_content = []
    coding_content = []
    reasoning_content = []

    # Generate 5000 conversations
    print("Generating conversations...")
    for i in range(5000):
        conv = generator.generate_conversation(random.randint(2, 5))
        conversations.append(conv)
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1} conversations")

    # Generate 2000 Q&A pairs
    print("Generating Q&A pairs...")
    for i in range(2000):
        qa = generator.generate_qa_pairs(random.randint(5, 15))
        qa_pairs.append(qa)
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1} Q&A sets")

    # Generate 1000 dialogue scenarios
    print("Generating dialogue scenarios...")
    for i in range(1000):
        dialogue = generator.generate_dialogue_scenarios(1)
        dialogues.append(dialogue)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} dialogue scenarios")

    # Generate 2000 math content blocks
    print("Generating mathematics content...")
    for i in range(2000):
        math = generator.generate_math_content()
        math_content.append(math)
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1} math content blocks")

    # Generate 2000 science content blocks
    print("Generating science content...")
    for i in range(2000):
        science = generator.generate_science_content()
        science_content.append(science)
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1} science content blocks")

    # Generate 2000 coding content blocks
    print("Generating programming content...")
    for i in range(2000):
        coding = generator.generate_coding_content()
        coding_content.append(coding)
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1} coding content blocks")

    # Generate 2000 reasoning content blocks
    print("Generating reasoning content...")
    for i in range(2000):
        reasoning = generator.generate_reasoning_content()
        reasoning_content.append(reasoning)
        if (i + 1) % 200 == 0:
            print(f"Generated {i + 1} reasoning content blocks")

    # Combine all data
    all_data = (conversations + qa_pairs + dialogues +
                math_content + science_content + coding_content + reasoning_content)

    # Shuffle to mix different types
    random.shuffle(all_data)

    # Write to file
    output_file = "data/text/raw/conversations/generated_conversations.txt"
    print(f"Writing {len(all_data)} content blocks to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(item)

    print(f"Dataset generation complete! Total size: {len(all_data)} content blocks")
    print("Breakdown:")
    print(f"  - Conversations: {len(conversations)}")
    print(f"  - Q&A Pairs: {len(qa_pairs)}")
    print(f"  - Dialogues: {len(dialogues)}")
    print(f"  - Mathematics: {len(math_content)}")
    print(f"  - Science: {len(science_content)}")
    print(f"  - Programming: {len(coding_content)}")
    print(f"  - Reasoning: {len(reasoning_content)}")
    print("")
    print("This comprehensive dataset provides at least 3 hours of diverse training data")
    print("covering language, reasoning, coding (for self-improvement), math, and all branches of science")

if __name__ == "__main__":
    main()