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

    print("Generating conversational datasets for HRM training...")

    # Generate large dataset
    conversations = []
    qa_pairs = []
    dialogues = []

    # Generate 1000 conversations
    print("Generating conversations...")
    for i in range(1000):
        conv = generator.generate_conversation(random.randint(2, 5))
        conversations.append(conv)
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} conversations")

    # Generate 5000 Q&A pairs
    print("Generating Q&A pairs...")
    for i in range(500):
        qa = generator.generate_qa_pairs(random.randint(5, 15))
        qa_pairs.append(qa)
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1} Q&A sets")

    # Generate 200 dialogue scenarios
    print("Generating dialogue scenarios...")
    for i in range(200):
        dialogue = generator.generate_dialogue_scenarios(1)
        dialogues.append(dialogue)
        if (i + 1) % 20 == 0:
            print(f"Generated {i + 1} dialogue scenarios")

    # Combine all data
    all_data = conversations + qa_pairs + dialogues

    # Shuffle to mix different types
    random.shuffle(all_data)

    # Write to file
    output_file = "data/text/raw/conversations/generated_conversations.txt"
    print(f"Writing {len(all_data)} conversation blocks to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(item)

    print(f"Dataset generation complete! Total size: {len(all_data)} conversation blocks")
    print("This should provide sufficient data for 3+ hours of HRM character-level training")

if __name__ == "__main__":
    main()