import unittest

import numpy as np

from voxck.face import FaceRenderer


class SilentFaceTests(unittest.TestCase):
    def renderer(self, idle_frame=1):
        # Distinct source frames model a clip whose opening still contains speech.
        face = FaceRenderer.__new__(FaceRenderer)
        face.frames = [np.full((16, 16, 3), value, np.uint8) for value in (20, 100, 220)]
        face.view = (0, 0, 16, 16)
        face.out_w = face.out_h = 16
        face.jpeg_quality = 80
        face.idle_frame = idle_frame
        face._cursor = 0
        return face

    def test_idle_uses_selected_neutral_frame_instead_of_source_movie(self):
        face = self.renderer()
        face._prepare_idle()
        neutral = face.jpeg(face._view(face.frames[1]))
        self.assertNotEqual(neutral, face.jpeg(face._view(face.frames[0])))
        self.assertTrue(all(face.next_idle_jpeg() == neutral for _ in range(100)))

    def test_idle_stays_neutral_after_talking_and_does_not_advance_talk_cursor(self):
        face = self.renderer()
        face._prepare_idle()
        neutral = face.next_idle_jpeg()
        face._cursor = 27
        for _ in range(100):
            self.assertEqual(face.next_idle_jpeg(), neutral)
        self.assertEqual(face._cursor, 27)

    def test_invalid_neutral_frame_cannot_silently_select_talking_opening(self):
        for index in (-1, 3, True, 1.5):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.renderer(index)._prepare_idle()


if __name__ == "__main__":
    unittest.main()
