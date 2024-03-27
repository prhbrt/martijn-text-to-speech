from typing import Tuple

import re
import numpy as np
from app.pipelines import Pipeline


class TextToSpeechPipeline(Pipeline):
    def __init__(self, model_id: str):
        # IMPLEMENT_THIS
        assert model_id == "notes"
        self.base_frequency = 440.0
        self.l = 0.05
        self.sample_rate = 44100

        self.notes = [
            'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E',
            'F', 'F#', 'G', 'G#', '^A', '^A#', '^B',
            '^B#', '^C', '^C#', '^D', '^D#', '^E', '^E#'
        ]
        self.notes_exp = re.compile(
            '(?: \d+|' +
            '|'.join(re.escape(x) + '\d+' for x in sorted(self.notes, key=len, reverse=True)) +
            ')'
        )
        self.note_exp = re.compile('(?P<note>[A-G^# ]+)(?P<duration>\d+)')

        self.frequencies = self.base_frequency * 2 ** (np.arange(len(self.notes)) / 12.0)

        self.note_to_frequency = dict(zip(self.notes, self.frequencies))
        self.note_to_frequency[' '] = 0


    def get_notes(self, str):
        return [
            (note.group('note'), int(note.group('duration')) * self.l)
            for note in self.notes_exp.findall(str)
            for note in [self.note_exp.match(note)]
        ]


    def note(self, frequency, duration, amplitude=1):
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        return amplitude * np.sin(2 * np.pi * frequency * t) * np.exp(-t)


    def __call__(self, inputs: str) -> Tuple[np.array, int]:
        """
        Args:
            inputs (:obj:`str`):
                The text to generate audio from
        Return:
            A :obj:`np.array` and a :obj:`int`: The raw waveform as a numpy array, and the sampling rate as an int.
        """
        # IMPLEMENT_THIS
        samples = np.concatenate([
            self.note(self.note_to_frequency[note], duration, np.iinfo(np.int16).max).astype(int) # weird clipping if both are `np.int16`.
            for note, duration in self.get_notes(inputs)
        ])

        return samples, self.sample_rate
