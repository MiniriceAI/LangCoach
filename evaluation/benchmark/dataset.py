"""
Benchmark Dataset for LangCoach Evaluation

This module provides a fixed dataset of 100 samples for benchmarking:
- User inputs for conversation testing
- Text samples for TTS evaluation
- Reference data for STT evaluation

The dataset is designed to be reproducible and supports running only the first n samples.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    id: int
    # User input text (simulates what user would say)
    user_input: str
    # Expected conversation context/scenario
    scenario: str
    # Difficulty level
    difficulty: str
    # Category for analysis
    category: str
    # Optional: reference audio path for STT testing
    audio_path: Optional[str] = None
    # Optional: expected transcription for STT accuracy
    expected_transcription: Optional[str] = None
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkDataset:
    """
    Benchmark dataset manager.

    Provides 100 fixed samples for reproducible evaluation.
    Supports loading a subset of samples for quick testing.
    """

    DATASET_FILE = "benchmark_samples.json"

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the benchmark dataset.

        Args:
            data_dir: Directory containing benchmark data.
                     Defaults to evaluation/benchmark/data
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._samples: List[BenchmarkSample] = []
        self._load_or_create_dataset()

    def _load_or_create_dataset(self):
        """Load existing dataset or create default one."""
        dataset_path = self.data_dir / self.DATASET_FILE

        if dataset_path.exists():
            self._load_dataset(dataset_path)
        else:
            self._create_default_dataset()
            self._save_dataset(dataset_path)

    def _load_dataset(self, path: Path):
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._samples = [
            BenchmarkSample(**sample) for sample in data['samples']
        ]

    def _save_dataset(self, path: Path):
        """Save dataset to JSON file."""
        data = {
            'version': '1.0',
            'total_samples': len(self._samples),
            'samples': [asdict(s) for s in self._samples]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _create_default_dataset(self):
        """Create the default 100-sample benchmark dataset."""
        samples = []
        sample_id = 0

        # Category 1: Job Interview (25 samples)
        job_interview_inputs = [
            "Hi, I'm here for the software engineer interview.",
            "I have five years of experience in Python development.",
            "My biggest strength is problem-solving.",
            "I left my previous job to seek new challenges.",
            "I'm looking for a salary around 150k.",
            "I can start in two weeks after giving notice.",
            "Yes, I'm comfortable with remote work.",
            "I prefer working in agile teams.",
            "My weakness is sometimes being too detail-oriented.",
            "I handled a difficult project by breaking it into smaller tasks.",
            "I'm passionate about clean code and best practices.",
            "I've led a team of three developers before.",
            "I'm familiar with AWS and cloud deployment.",
            "I enjoy mentoring junior developers.",
            "My long-term goal is to become a tech lead.",
            "I've worked with React and Vue.js frameworks.",
            "I handle stress by prioritizing tasks effectively.",
            "I'm interested in your company's AI initiatives.",
            "I've contributed to several open source projects.",
            "I believe in continuous learning and improvement.",
            "I can work well both independently and in teams.",
            "I've experience with microservices architecture.",
            "I'm comfortable with code reviews and pair programming.",
            "I've dealt with tight deadlines successfully.",
            "Do you have any questions about my background?",
        ]

        for text in job_interview_inputs:
            samples.append(BenchmarkSample(
                id=sample_id,
                user_input=text,
                scenario="job_interview",
                difficulty="medium",
                category="job_interview",
                expected_transcription=text,
            ))
            sample_id += 1

        # Category 2: Hotel Check-in (25 samples)
        hotel_inputs = [
            "Hello, I have a reservation under the name Smith.",
            "I booked a room for three nights.",
            "Do you have any rooms with a sea view?",
            "What time is breakfast served?",
            "Is there WiFi in the rooms?",
            "Can I get a late checkout tomorrow?",
            "Where is the nearest restaurant?",
            "I'd like to request extra towels please.",
            "Is the gym open 24 hours?",
            "Can you recommend any local attractions?",
            "I need a wake-up call at 7 AM.",
            "Is parking included in the room rate?",
            "Can I store my luggage after checkout?",
            "The air conditioning isn't working properly.",
            "I'd like to extend my stay by one night.",
            "Do you have a business center?",
            "Can I get a room on a higher floor?",
            "Is there a shuttle to the airport?",
            "I need to change my room, it's too noisy.",
            "What's the checkout time?",
            "Can I pay with a credit card?",
            "Is there a minibar in the room?",
            "I'd like to order room service.",
            "Can you call a taxi for me?",
            "Thank you for your help.",
        ]

        for text in hotel_inputs:
            samples.append(BenchmarkSample(
                id=sample_id,
                user_input=text,
                scenario="hotel_checkin",
                difficulty="primary",
                category="hotel_checkin",
                expected_transcription=text,
            ))
            sample_id += 1

        # Category 3: Renting (25 samples)
        renting_inputs = [
            "I'm looking for a two-bedroom apartment.",
            "What's the monthly rent for this place?",
            "Are utilities included in the rent?",
            "Is there a security deposit required?",
            "How long is the lease term?",
            "Can I see the apartment today?",
            "Is the neighborhood safe?",
            "Are pets allowed in this building?",
            "When would the apartment be available?",
            "Is there in-unit laundry?",
            "How much is the parking fee?",
            "Are there any move-in specials?",
            "What's included in the lease?",
            "Can I paint the walls?",
            "Is there central air conditioning?",
            "How do I submit maintenance requests?",
            "What's the policy on subletting?",
            "Are there any noise restrictions?",
            "Is renters insurance required?",
            "Can I install my own internet?",
            "What happens if I need to break the lease?",
            "Is there a gym in the building?",
            "How is the water pressure?",
            "Are there any upcoming rent increases?",
            "I'd like to apply for this apartment.",
        ]

        for text in renting_inputs:
            samples.append(BenchmarkSample(
                id=sample_id,
                user_input=text,
                scenario="renting",
                difficulty="medium",
                category="renting",
                expected_transcription=text,
            ))
            sample_id += 1

        # Category 4: Salary Negotiation (25 samples)
        salary_inputs = [
            "I'd like to discuss my compensation package.",
            "Based on my research, the market rate is higher.",
            "I believe my experience justifies a higher salary.",
            "Can we discuss the bonus structure?",
            "What about stock options or equity?",
            "I'm looking for a 15% increase.",
            "My current offer is below my expectations.",
            "Can we revisit this in six months?",
            "What's the typical raise percentage here?",
            "I have another offer with better compensation.",
            "Is there flexibility in the base salary?",
            "What benefits are included in the package?",
            "Can we negotiate the signing bonus?",
            "I'd like more vacation days instead.",
            "What's the path to promotion here?",
            "How often are performance reviews conducted?",
            "Is remote work an option for this role?",
            "Can we discuss professional development budget?",
            "What's the 401k matching policy?",
            "I need to think about this offer.",
            "Can you put this offer in writing?",
            "What's the deadline for my decision?",
            "I'm excited about this opportunity.",
            "Let me discuss this with my family.",
            "I accept the offer. Thank you.",
        ]

        for text in salary_inputs:
            samples.append(BenchmarkSample(
                id=sample_id,
                user_input=text,
                scenario="salary_negotiation",
                difficulty="advanced",
                category="salary_negotiation",
                expected_transcription=text,
            ))
            sample_id += 1

        self._samples = samples

    def get_samples(self, n: Optional[int] = None) -> List[BenchmarkSample]:
        """
        Get benchmark samples.

        Args:
            n: Number of samples to return. If None, returns all samples.
               Useful for quick testing with fewer samples.

        Returns:
            List of BenchmarkSample objects
        """
        if n is None:
            return self._samples.copy()
        return self._samples[:n]

    def get_samples_by_scenario(self, scenario: str) -> List[BenchmarkSample]:
        """Get samples filtered by scenario."""
        return [s for s in self._samples if s.scenario == scenario]

    def get_samples_by_difficulty(self, difficulty: str) -> List[BenchmarkSample]:
        """Get samples filtered by difficulty."""
        return [s for s in self._samples if s.difficulty == difficulty]

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, idx: int) -> BenchmarkSample:
        return self._samples[idx]


# Convenience function
def load_benchmark(n: Optional[int] = None) -> List[BenchmarkSample]:
    """
    Load benchmark samples.

    Args:
        n: Number of samples to load. If None, loads all 100 samples.

    Returns:
        List of BenchmarkSample objects
    """
    dataset = BenchmarkDataset()
    return dataset.get_samples(n)
