"""Observed display cadence and physical playback rate use distinct units."""
import unittest

from rigid_coupling.viewer import PlaybackStats


class PlaybackStatsChecks(unittest.TestCase):
    def test_variable_frames_report_observed_fps_and_physical_speed(self):
        stats = PlaybackStats(.002, interval=.04)
        stats.record(.01, 1)
        self.assertIsNone(stats.fps)
        stats.record(.03, 3)
        self.assertAlmostEqual(stats.fps, 50.)
        self.assertAlmostEqual(stats.speed, .2)

    def test_paused_frames_keep_display_fps_and_report_zero_motion(self):
        stats = PlaybackStats(.002, interval=.04)
        stats.record(.02, 0)
        stats.record(.02, 0)
        self.assertAlmostEqual(stats.fps, 50.)
        self.assertEqual(stats.speed, 0.)
        stats.reset()
        self.assertIsNone(stats.fps)
        self.assertEqual(stats.speed, 0.)


if __name__ == "__main__":
    unittest.main()
