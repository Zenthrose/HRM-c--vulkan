#!/usr/bin/env python3
"""
ArXiv Content Downloader and Processor for HRM Training
Downloads and processes arXiv papers for character-level language training
"""

import os
import sys
import json
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
import re
from collections import Counter

class ArXivDownloader:
    """Download and process arXiv content for HRM training"""

    def __init__(self, max_papers=100, categories=None):
        self.max_papers = max_papers
        self.categories = categories or [
            'cs.AI', 'cs.LG', 'cs.CL', 'cs.NE',  # Computer Science
            'stat.ML', 'math.ST',  # Statistics & Math
            'physics.comp-ph', 'quant-ph'  # Physics/Quantum
        ]
        self.base_url = "http://export.arxiv.org/api/query?"

    def download_recent_papers(self, days_back=7):
        """Download recent papers from arXiv"""
        print(f"📥 Downloading recent arXiv papers (last {days_back} days)...")

        all_papers = []

        for category in self.categories:
            print(f"  Downloading from category: {category}")

            # Query parameters
            params = {
                'search_query': f'cat:{category}',
                'sortBy': 'submittedDate',
                'sortOrder': 'descending',
                'max_results': min(50, self.max_papers // len(self.categories)),
                'start': 0
            }

            try:
                url = self.base_url + urlencode(params)
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Parse XML response
                root = ET.fromstring(response.content)

                # Extract paper information
                namespace = {'arxiv': 'http://www.w3.org/2005/Atom',
                           'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}

                entries = root.findall('arxiv:entry', namespace)
                print(f"    Found {len(entries)} papers in {category}")

                for entry in entries:
                    paper = self._extract_paper_info(entry, namespace)
                    if paper:
                        all_papers.append(paper)

            except Exception as e:
                print(f"    Error downloading from {category}: {e}")
                continue

        print(f"✅ Downloaded {len(all_papers)} total papers")
        return all_papers

    def _extract_paper_info(self, entry, namespace):
        """Extract paper information from XML entry"""
        try:
            # Get basic info
            title_elem = entry.find('arxiv:title', namespace)
            abstract_elem = entry.find('arxiv:summary', namespace)
            authors_elem = entry.findall('arxiv:author', namespace)
            published_elem = entry.find('arxiv:published', namespace)
            id_elem = entry.find('arxiv:id', namespace)

            if not all([title_elem is not None, abstract_elem is not None]):
                return None

            # Extract text content
            title = title_elem.text.strip() if title_elem.text else ""
            abstract = abstract_elem.text.strip() if abstract_elem.text else ""

            # Clean up text
            title = self._clean_text(title)
            abstract = self._clean_text(abstract)

            # Extract authors
            authors = []
            for author_elem in authors_elem:
                name_elem = author_elem.find('arxiv:name', namespace)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Extract date
            published_date = ""
            if published_elem is not None and published_elem.text:
                published_date = published_elem.text[:10]  # YYYY-MM-DD

            # Extract arXiv ID
            arxiv_id = ""
            if id_elem is not None and id_elem.text:
                arxiv_id = id_elem.text.split('/')[-1]

            return {
                'id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'published_date': published_date,
                'full_text': f"{title}\n\n{abstract}"
            }

        except Exception as e:
            print(f"Error extracting paper info: {e}")
            return None

    def _clean_text(self, text):
        """Clean and normalize text content"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove LaTeX commands (simplified)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\$[^$]*\$', '', text)  # Remove inline math

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        return text

    def save_papers(self, papers, output_dir="./data/arxiv"):
        """Save downloaded papers to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save individual papers
        papers_dir = Path(output_dir) / "papers"
        papers_dir.mkdir(exist_ok=True)

        for i, paper in enumerate(papers):
            filename = f"paper_{i:04d}_{paper['id']}.txt"
            filepath = papers_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {paper['title']}\n")
                f.write(f"Authors: {', '.join(paper['authors'])}\n")
                f.write(f"Date: {paper['published_date']}\n")
                f.write(f"ArXiv ID: {paper['id']}\n")
                f.write(f"\n{paper['full_text']}\n")

        # Save combined corpus
        corpus_file = Path(output_dir) / "arxiv_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(f"{paper['full_text']}\n\n")

        # Save metadata
        metadata_file = Path(output_dir) / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_papers': len(papers),
                'categories': self.categories,
                'download_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_characters': sum(len(p['full_text']) for p in papers),
                'papers': papers
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(papers)} papers to {output_dir}")
        print(f"📊 Total characters: {sum(len(p['full_text']) for p in papers):,}")

        return str(corpus_file)

class ArXivProcessor:
    """Process arXiv content for HRM training"""

    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def analyze_corpus(self):
        """Analyze the arXiv corpus characteristics"""
        print("📊 Analyzing arXiv corpus...")

        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Basic statistics
        total_chars = len(text)
        total_words = len(text.split())
        unique_chars = len(set(text))

        # Character frequency analysis
        char_counts = Counter(text)
        most_common_chars = char_counts.most_common(20)

        # Word analysis
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        most_common_words = word_counts.most_common(20)

        print(f"📈 Corpus Statistics:")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Total words: {total_words:,}")
        print(f"   Unique characters: {unique_chars}")
        print(f"   Average word length: {sum(len(w) for w in words) / len(words):.1f}")

        print(f"\n🔤 Top 20 Characters:")
        for char, count in most_common_chars:
            char_repr = repr(char) if char.isprintable() and char not in '\n\t' else f"\\x{ord(char):02x}"
            percentage = (count / total_chars) * 100
            print("6s")

        print(f"\n📝 Top 20 Words:")
        for word, count in most_common_words:
            percentage = (count / len(words)) * 100
            print("12s")

        return {
            'total_chars': total_chars,
            'total_words': total_words,
            'unique_chars': unique_chars,
            'char_distribution': dict(most_common_chars),
            'word_distribution': dict(most_common_words)
        }

    def prepare_training_data(self, output_dir="./data/text/processed"):
        """Prepare arXiv data for HRM training"""
        print("🎯 Preparing training data for HRM...")

        # Read corpus
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            corpus_text = f.read()

        # Split into training/validation sets
        split_point = int(len(corpus_text) * 0.9)
        train_text = corpus_text[:split_point]
        val_text = corpus_text[split_point:]

        # Save processed data
        os.makedirs(output_dir, exist_ok=True)

        train_file = Path(output_dir) / "arxiv_training_corpus.txt"
        val_file = Path(output_dir) / "arxiv_validation_corpus.txt"

        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(train_text)

        with open(val_file, 'w', encoding='utf-8') as f:
            f.write(val_text)

        print(f"✅ Training data: {len(train_text):,} characters")
        print(f"✅ Validation data: {len(val_text):,} characters")
        print(f"📁 Saved to: {output_dir}")

        return str(train_file), str(val_file)

def main():
    """Main function to download and process arXiv content"""
    print("🚀 ArXiv Content Processor for HRM Training")
    print("=" * 50)

    # Configuration
    max_papers = 200  # Download 200 papers
    output_dir = "./data/arxiv"

    # Download papers
    downloader = ArXivDownloader(max_papers=max_papers)
    papers = downloader.download_recent_papers(days_back=30)  # Last 30 days

    if not papers:
        print("❌ No papers downloaded!")
        return

    # Save papers
    corpus_file = downloader.save_papers(papers, output_dir)

    # Analyze corpus
    processor = ArXivProcessor(corpus_file)
    stats = processor.analyze_corpus()

    # Prepare training data
    train_file, val_file = processor.prepare_training_data()

    print("\n🎯 HRM Training Data Ready!")
    print(f"   Training file: {train_file}")
    print(f"   Validation file: {val_file}")
    print(f"   Total papers: {len(papers)}")
    print(f"   Total characters: {stats['total_chars']:,}")
    print(f"   Unique characters: {stats['unique_chars']}")

    # Create training configuration
    config = {
        "dataset_name": "arxiv_recent_papers",
        "total_papers": len(papers),
        "total_characters": stats['total_chars'],
        "unique_characters": stats['unique_chars'],
        "training_file": train_file,
        "validation_file": val_file,
        "categories": downloader.categories,
        "download_date": time.strftime('%Y-%m-%d %H:%M:%S')
    }

    config_file = Path(output_dir) / "training_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"   Config file: {config_file}")

    return train_file

if __name__ == "__main__":
    try:
        training_file = main()
        if training_file:
            print(f"\n✅ Ready to train HRM on arXiv content: {training_file}")
        else:
            print("\n❌ Failed to prepare training data")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)