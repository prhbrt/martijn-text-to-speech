import os
from unittest import TestCase, skipIf

from api_inference_community.validation import ffmpeg_read
from app.main import ALLOWED_TASKS
from starlette.testclient import TestClient
from tests.test_api import TESTABLE_MODELS


@skipIf(
    "text-to-speech" not in ALLOWED_TASKS,
    "text-to-speech not implemented",
)
class TextToSpeechTestCase(TestCase):
    def setUp(self):
        model_id = TESTABLE_MODELS["text-to-speech"]
        self.old_model_id = os.getenv("MODEL_ID")
        self.old_task = os.getenv("TASK")
        os.environ["MODEL_ID"] = model_id
        os.environ["TASK"] = "text-to-speech"
        from app.main import app

        self.app = app

    @classmethod
    def setUpClass(cls):
        from app.main import get_pipeline

        get_pipeline.cache_clear()

    def tearDown(self):
        if self.old_model_id is not None:
            os.environ["MODEL_ID"] = self.old_model_id
        else:
            del os.environ["MODEL_ID"]
        if self.old_task is not None:
            os.environ["TASK"] = self.old_task
        else:
            del os.environ["TASK"]

    def test_simple(self):
        with TestClient(self.app) as client:
            response = client.post("/", json={"inputs": "C2D2E2C2C2D2E2C2E2F2G2 2E2F2G2 2G1^A1G1F1E2C2G1^A1G1F1E2C2C2G2C2 2C2G2C2 2"})

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(response.headers["content-type"], "audio/flac")
        audio = ffmpeg_read(response.content, 16000)
        self.assertEqual(len(audio.shape), 1)
        self.assertGreater(audio.shape[0], 1000)
        self.assertGreater(audio.max(), 0)
        self.assertLess(audio.min(), 0)

    def test_malformed_input(self):
        with TestClient(self.app) as client:
            response = client.post("/", data=b"\xc3\x28")

        self.assertEqual(
            response.status_code,
            400,
        )
        self.assertEqual(
            response.content,
            b'{"error":"\'utf-8\' codec can\'t decode byte 0xc3 in position 0: invalid continuation byte"}',
        )
