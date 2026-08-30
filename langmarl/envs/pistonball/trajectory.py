"""Trajectory formatting for the Pistonball environment."""

from typing import Dict


class PistonballTrajectoryFormatter:
    """
    Format episode trajectories for LLM evaluation in Pistonball.

    All paradigms use the same format:
    - Global state (from format_global_state in episode_generator.py)
    - Each agent's action
    - Current cumulative reward
    """

    ACTION_NAMES = {
        0: "retract_down",
        1: "stay",
        2: "push_up"
    }

    @staticmethod
    def format_trajectory(episode: Dict) -> str:
        """
        Format trajectory for LLM evaluation.

        Uses the global state text stored in each transition,
        along with agent actions and cumulative reward.

        Args:
            episode: Episode dict containing:
                - transitions: List of transition dicts, each with:
                    - timestep: int
                    - global_state: dict with 'text' key containing formatted state
                    - agents: dict mapping agent_name -> action info
                    - instant_reward: float
                    - cumulative_reward: float
                - total_reward: float

        Returns:
            Formatted trajectory string
        """
        transitions = episode['transitions']
        total_reward = episode['total_reward']
        num_pistons = episode.get('num_pistons', 20)

        lines = []
        lines.append("=== EPISODE TRAJECTORY ===")
        lines.append(f"Total Pistons: {num_pistons}")
        lines.append(f"Total Reward: {total_reward:.2f}")
        lines.append("")

        for trans in transitions:
            timestep = trans['timestep']

            # Sample every 10 timesteps (0, 10, 20, ...)
            if timestep % 10 != 0:
                continue

            instant_reward = trans['instant_reward']
            cumulative_reward = trans['cumulative_reward']

            # Get global state text
            global_state = trans.get('global_state', {})
            global_state_text = global_state.get('text', '[No global state available]')

            # Get agent actions
            agents_info = trans.get('agents', {})

            lines.append(f"--- Timestep {timestep} ---")
            lines.append("")
            lines.append("[Global State]")
            lines.append(global_state_text)
            lines.append("")
            lines.append("[Actions]")

            for agent_name in sorted(agents_info.keys()):
                agent_info = agents_info[agent_name]
                action_name = agent_info.get('action_name', 'unknown')
                lines.append(f"  {agent_name}: {action_name}")

            lines.append("")
            lines.append(f"[Reward] instant={instant_reward:.4f}, cumulative={cumulative_reward:.4f}")
            lines.append("")

        lines.append(f"=== END TRAJECTORY (Total Reward: {total_reward:.2f}) ===")

        return "\n".join(lines)
