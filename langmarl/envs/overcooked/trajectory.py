"""Trajectory formatting for the Overcooked environment."""

from typing import Dict, List, Optional


class OvercookedTrajectoryFormatter:
    """Format episode trajectories for LLM evaluation."""

    @staticmethod
    def format_for_evaluation(
        episode: Dict,
        agent_id: Optional[str] = None
    ) -> str:
        """
        Format trajectory for Monte Carlo evaluation.

        Args:
            episode: Episode dict containing transitions and total_reward
            agent_id: '0' or '1' for individual view, None for global view

        Returns:
            Formatted trajectory string
        """
        transitions = episode['transitions']
        total_reward = episode['total_reward']

        formatted_lines = []
        formatted_lines.append(
            f"=== TRAJECTORY (Total Reward: {total_reward:.0f}) ===\n"
        )

        for trans in transitions:
            timestep = trans['timestep']
            instant_reward = trans['instant_reward']

            # Note: Rewards are sparse - only non-zero when soup is delivered
            reward_note = " (Soup delivered!)" if instant_reward > 0 else " (No reward - sparse)"

            if agent_id is None:
                # Global view: show both agents
                agent0_info = trans['agents'].get('0', {})
                agent1_info = trans['agents'].get('1', {})

                obs0 = agent0_info.get('observation_to_gpt', 'N/A')
                action0 = agent0_info.get('parsed_ml_action', 'N/A')
                action1 = agent1_info.get('parsed_ml_action', 'N/A')

                # States are shared between agents, so we can use agent0's observation
                formatted_lines.append(
                    f"Timestep {timestep}:\n"
                    f"  State: {obs0}\n"
                    f"  Agent 0 Action: {action0}\n"
                    f"  Agent 1 Action: {action1}\n"
                    f"  Reward: {instant_reward:.1f}{reward_note}"
                )
            else:
                # Individual agent view
                if agent_id in trans['agents']:
                    agent_info = trans['agents'][agent_id]
                    observation = agent_info.get('observation_to_gpt', 'N/A')
                    ml_action = agent_info.get('parsed_ml_action', 'N/A')

                    formatted_lines.append(
                        f"Timestep {timestep}:\n"
                        f"  State: {observation}\n"
                        f"  Action: {ml_action}\n"
                        f"  Reward: {instant_reward:.1f}{reward_note}"
                    )

        formatted_lines.append(
            f"\n=== END TRAJECTORY (Total: {total_reward:.0f} points) ==="
        )

        return "\n\n".join(formatted_lines)

    @staticmethod
    def format_trajectory(episode: Dict, agent_id: Optional[str] = None) -> str:
        """Alias for format_for_evaluation (unified API)."""
        return OvercookedTrajectoryFormatter.format_for_evaluation(episode, agent_id)

    @staticmethod
    def format_for_credit_assignment(episode: Dict) -> str:
        """Format trajectory for credit assignment (shows both agents' actions globally)."""
        return OvercookedTrajectoryFormatter.format_for_evaluation(episode, agent_id=None)

    @staticmethod
    def format_trajectory_summary(episodes: List[Dict]) -> str:
        """
        Format a summary of multiple episodes.

        Args:
            episodes: List of episode dicts

        Returns:
            Summary string
        """
        if not episodes:
            return "No episodes to summarize"

        rewards = [ep['total_reward'] for ep in episodes]
        avg_reward = sum(rewards) / len(rewards)

        summary_lines = [
            "=== EPISODE BATCH SUMMARY ===",
            f"Number of episodes: {len(episodes)}",
            f"Rewards: {rewards}",
            f"Average reward: {avg_reward:.1f}",
            f"Min reward: {min(rewards):.0f}",
            f"Max reward: {max(rewards):.0f}",
        ]

        return "\n".join(summary_lines)
