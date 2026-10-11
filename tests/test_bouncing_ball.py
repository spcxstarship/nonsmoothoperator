import unittest

from packages.dynamics.bouncing_ball import BouncingBall


class BouncingBallTests(unittest.TestCase):
    def test_impact_has_two_velocity_states_and_no_height_jump(self) -> None:
        ball = BouncingBall(initial_height=1.0, restitution=0.8)
        before = ball.state_at(ball.impact_time, side="pre")
        after = ball.state_at(ball.impact_time, side="post")

        self.assertAlmostEqual(before.height, 0.0)
        self.assertAlmostEqual(after.height, 0.0)
        self.assertAlmostEqual(after.velocity, -ball.restitution * before.velocity)

    def test_energy_is_conserved_in_flight_and_drops_at_impact(self) -> None:
        ball = BouncingBall()

        def energy(time: float, side: str = "post") -> float:
            state = ball.state_at(time, side=side)
            return 0.5 * state.velocity**2 + ball.gravity * state.height

        self.assertAlmostEqual(energy(ball.impact_time / 2), energy(0))
        self.assertAlmostEqual(
            energy(ball.impact_time),
            ball.restitution**2 * energy(ball.impact_time, side="pre"),
        )
        self.assertAlmostEqual(
            energy((ball.impact_time + ball.next_impact_time) / 2),
            energy(ball.impact_time),
        )

    def test_invalid_parameters_and_out_of_scope_time(self) -> None:
        for kwargs in (
            {"initial_height": 0},
            {"gravity": 0},
            {"restitution": -0.1},
            {"restitution": 1},
            {"applied_acceleration": 9.81},
        ):
            with self.assertRaises(ValueError):
                BouncingBall(**kwargs)

        ball = BouncingBall()
        with self.assertRaises(ValueError):
            ball.state_at(ball.next_impact_time + 0.01)

    def test_constant_actuation_changes_impact_time(self) -> None:
        downward = BouncingBall(applied_acceleration=-2.0)
        neutral = BouncingBall()
        upward = BouncingBall(applied_acceleration=2.0)
        self.assertLess(downward.impact_time, neutral.impact_time)
        self.assertLess(neutral.impact_time, upward.impact_time)
        self.assertAlmostEqual(
            upward.state_at(upward.impact_time, side="pre").velocity,
            -upward.impact_speed,
        )


if __name__ == "__main__":
    unittest.main()
